"""MiniLMv2-style token-relation distillation on pooled backbone states.

MiniLMv2 (Wang et al. 2021) distills the *relations* between positions —
softmax-normalized pairwise similarities — instead of raw activations, so
teacher and student may differ in hidden size, depth, and head count. Prism
adapts the idea to the word-aligned pooled backbone representations that
every ``TokenTagger`` produces before its task heads: that boundary is
architecture-agnostic (no hooks into the pinned custom backbone code) and
tokenizer-agnostic (both models describe the same word sequence), and the
multi-head split below restores MiniLMv2's relation heads on top of it.

The loss is the KL divergence between the teacher's and the student's
relation distributions, averaged over relation heads and valid tokens. It
is intended as an auxiliary objective on gold batches, where the frozen
relation teacher runs one forward pass per batch.
"""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationDistillationPolicy:
    """Weight and relation-head configuration of the auxiliary loss."""

    weight: float
    relation_head_count: int = 8

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise ValueError("Relation-distillation weight must be positive.")
        if self.relation_head_count <= 0:
            raise ValueError("Relation head count must be positive.")


def _relation_scores(
    hidden_states: torch.Tensor,
    *,
    relation_head_count: int,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, token_count, hidden_size = hidden_states.shape
    if hidden_size % relation_head_count != 0:
        raise ValueError(
            "Hidden size must be divisible by the relation head count: "
            f"{hidden_size} % {relation_head_count} != 0."
        )
    head_size = hidden_size // relation_head_count
    heads = hidden_states.view(
        batch_size,
        token_count,
        relation_head_count,
        head_size,
    ).transpose(1, 2)
    scores = heads @ heads.transpose(-1, -2) / math.sqrt(head_size)
    # Finite mask value: -inf would turn 0 * (-inf - -inf) into NaN in the
    # KL term below, so invalid key positions get a dominating finite score.
    masked_score = torch.finfo(scores.dtype).min / 2
    return scores.masked_fill(
        ~token_mask[:, None, None, :],
        masked_score,
    )


def compute_token_relation_loss(
    *,
    student_hidden_states: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    token_mask: torch.Tensor,
    relation_head_count: int,
) -> torch.Tensor:
    """KL divergence between teacher and student token-relation distributions.

    Both inputs are word-aligned pooled backbone states of shape
    ``[batch, token, hidden]`` over the same token sequence; hidden sizes may
    differ. Invalid (padding) tokens neither attend nor contribute.
    """

    if student_hidden_states.shape[:2] != teacher_hidden_states.shape[:2]:
        raise ValueError(
            "Student and teacher must describe the same token sequence: "
            f"{tuple(student_hidden_states.shape[:2])} != "
            f"{tuple(teacher_hidden_states.shape[:2])}."
        )
    if token_mask.shape != student_hidden_states.shape[:2]:
        raise ValueError("Token mask must match the token dimensions.")
    valid_token_count = int(token_mask.sum().item())
    if valid_token_count == 0:
        raise ValueError("Relation loss requires at least one valid token.")

    student_scores = _relation_scores(
        student_hidden_states,
        relation_head_count=relation_head_count,
        token_mask=token_mask,
    )
    with torch.no_grad():
        teacher_scores = _relation_scores(
            teacher_hidden_states,
            relation_head_count=relation_head_count,
            token_mask=token_mask,
        )
        teacher_probabilities = F.softmax(teacher_scores, dim=-1)
        teacher_log_probabilities = F.log_softmax(teacher_scores, dim=-1)

    student_log_probabilities = F.log_softmax(student_scores, dim=-1)
    divergence = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    divergence = divergence.masked_fill(~token_mask[:, None, :], 0.0)
    return divergence.sum() / (valid_token_count * relation_head_count)
