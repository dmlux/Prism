from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextualizedSubwordBatch:
    hidden_states: Tensor

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 3:
            raise ValueError("Contextualized hidden states must have three dimensions.")
        if not self.hidden_states.is_floating_point():
            raise ValueError(
                "Contextualized hidden states must use a floating-point dtype."
            )
        if not torch.isfinite(self.hidden_states).all().item():
            raise ValueError(
                "Contextualized hidden states must contain only finite values."
            )
        if any(dimension <= 0 for dimension in self.hidden_states.shape):
            raise ValueError("Contextualized hidden-state dimensions must be positive.")

    @property
    def batch_size(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def max_subword_count(self) -> int:
        return self.hidden_states.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.hidden_states.shape[2]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextualizedTokenBatch:
    hidden_states: Tensor
    token_mask: Tensor

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 3:
            raise ValueError("Contextualized token states must have three dimensions.")
        if not self.hidden_states.is_floating_point():
            raise ValueError(
                "Contextualized token states must use a floating-point dtype."
            )
        if self.token_mask.shape != self.hidden_states.shape[:2]:
            raise ValueError("Token mask shape must match contextualized token states.")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("Token mask use torch.bool.")

    @property
    def batch_size(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def max_token_count(self) -> int:
        return self.hidden_states.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.hidden_states.shape[2]


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskLogits:
    upos_logits: Tensor
    morphology_logits: tuple[Tensor, ...]
    lemma_rule_logits: Tensor

    def __post_init__(self) -> None:
        if not self.morphology_logits:
            raise ValueError(
                "Token task logits must contain morphology feature logits."
            )

        all_logits = (
            self.upos_logits,
            *self.morphology_logits,
            self.lemma_rule_logits,
        )

        if any(logits.ndim != 3 for logits in all_logits):
            raise ValueError("Token task logits must have three dimensions.")

        if any(not logits.is_floating_point() for logits in all_logits):
            raise ValueError("Token task logits must use floating-point dtypes.")

        token_dimensions = self.upos_logits.shape[:2]

        if any(logits.shape[:2] != token_dimensions for logits in all_logits):
            raise ValueError("Token task logits must share batch and token dimensions.")

    @property
    def batch_size(self) -> int:
        return self.upos_logits.shape[0]

    @property
    def max_token_count(self) -> int:
        return self.upos_logits.shape[1]

    @property
    def morphology_feature_count(self) -> int:
        return len(self.morphology_logits)
