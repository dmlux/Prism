import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from prism.modeling import TokenTaskLogits
from prism.training.losses import TokenTaskLosses, TokenTaskLossWeights


@dataclass(frozen=True, slots=True, kw_only=True)
class CombinedTokenTaskLosses:
    supervised_losses: TokenTaskLosses
    distillation_losses: TokenTaskLosses
    total_loss: torch.Tensor


def calculate_categorical_distillation_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    if token_mask.shape != student_logits.shape[:-1]:
        raise ValueError("Token mask must match the batch and token dimensions.")
    if temperature <= 0.0:
        raise ValueError("Temperature must be greater than zero.")

    selected_student_logits = student_logits[token_mask]
    selected_teacher_logits = teacher_logits[token_mask].detach()

    if selected_student_logits.numel() == 0:
        return student_logits.sum() * 0.0

    student_log_probabilities = F.log_softmax(
        selected_student_logits / temperature,
        dim=-1,
    )
    teacher_probabilities = F.softmax(
        selected_teacher_logits / temperature,
        dim=-1,
    )

    return F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature**2)


def calculate_binary_distillation_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
    positive_targets: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes.")
    if token_mask.shape != student_logits.shape[:-1]:
        raise ValueError("Token mask must match the batch and token dimensions.")
    if temperature <= 0.0:
        raise ValueError("Temperature must be greater than zero.")
    if (positive_targets is None) != (positive_weights is None):
        raise ValueError(
            "Positive targets and positive weights must be provided together."
        )

    student_binary_logits = torch.stack(
        (
            torch.zeros_like(student_logits),
            student_logits,
        ),
        dim=-1,
    )
    teacher_binary_logits = torch.stack(
        (
            torch.zeros_like(teacher_logits),
            teacher_logits,
        ),
        dim=-1,
    )

    student_log_probabilities = F.log_softmax(
        student_binary_logits / temperature,
        dim=-1,
    )
    teacher_probabilities = F.softmax(
        teacher_binary_logits.detach() / temperature,
        dim=-1,
    )
    per_label_losses = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)

    if positive_targets is not None and positive_weights is not None:
        if positive_targets.shape != student_logits.shape:
            raise ValueError("Positive targets must match the logit shape.")
        if positive_targets.dtype != torch.bool:
            raise ValueError("Positive targets must use the boolean data type.")
        if positive_weights.ndim != 1:
            raise ValueError("Positive weights must have one dimension.")
        if positive_weights.shape[0] != student_logits.shape[-1]:
            raise ValueError("Positive weights must match the label count.")
        if not positive_weights.is_floating_point():
            raise ValueError("Positive weights must be floating point.")
        if not torch.isfinite(positive_weights).all():
            raise ValueError("Positive weights must be finite.")
        if torch.any(positive_weights <= 0):
            raise ValueError("Positive weights must be positive.")

        broadcast_shape = (1,) * (student_logits.ndim - 1) + (-1,)
        per_label_weights = torch.where(
            positive_targets,
            positive_weights.reshape(broadcast_shape),
            torch.ones_like(student_logits),
        )
        per_label_losses = per_label_losses * per_label_weights

    label_mask = token_mask.unsqueeze(-1).expand_as(student_logits)
    selected_losses = per_label_losses.masked_select(label_mask)

    if selected_losses.numel() == 0:
        return student_logits.sum() * 0.0

    return selected_losses.mean()


def compute_token_task_distillation_loss(
    *,
    student_logits: TokenTaskLogits,
    teacher_logits: TokenTaskLogits,
    token_mask: torch.Tensor,
    lemma_rule_mask: torch.Tensor,
    temperature: float,
    morphology_targets: tuple[torch.Tensor, ...] | None = None,
    loss_weights: TokenTaskLossWeights | None = None,
) -> TokenTaskLosses:
    if (
        student_logits.morphology_feature_count
        != teacher_logits.morphology_feature_count
    ):
        raise ValueError(
            "Student and teacher must contain the same morphology features."
        )
    if lemma_rule_mask.shape != token_mask.shape:
        raise ValueError("Lemma-rule mask must match the token mask.")

    morphology_feature_count = student_logits.morphology_feature_count

    if loss_weights is None:
        if (
            morphology_targets is not None
            and len(morphology_targets) != morphology_feature_count
        ):
            raise ValueError(
                "Morphology targets must match the morphology feature count."
            )
        resolved_morphology_targets: tuple[torch.Tensor | None, ...] = (
            None,
        ) * morphology_feature_count
        morphology_positive_weights: tuple[torch.Tensor | None, ...] = (
            None,
        ) * morphology_feature_count
    else:
        if morphology_targets is None:
            raise ValueError(
                "Morphology targets are required when loss weights are provided."
            )
        if len(morphology_targets) != morphology_feature_count:
            raise ValueError(
                "Morphology targets must match the morphology feature count."
            )
        resolved_morphology_targets = morphology_targets
        if len(loss_weights.morphology_positive_weights) != morphology_feature_count:
            raise ValueError(
                "Morphology loss weights must match the morphology feature count."
            )
        morphology_positive_weights = loss_weights.morphology_positive_weights

    upos_loss = calculate_categorical_distillation_loss(
        student_logits=student_logits.upos_logits,
        teacher_logits=teacher_logits.upos_logits,
        token_mask=token_mask,
        temperature=temperature,
    )

    morphology_feature_losses = tuple(
        calculate_binary_distillation_loss(
            student_logits=student_feature_logits,
            teacher_logits=teacher_feature_logits,
            token_mask=token_mask,
            temperature=temperature,
            positive_targets=feature_targets,
            positive_weights=positive_weights,
        )
        for (
            student_feature_logits,
            teacher_feature_logits,
            feature_targets,
            positive_weights,
        ) in zip(
            student_logits.morphology_logits,
            teacher_logits.morphology_logits,
            resolved_morphology_targets,
            morphology_positive_weights,
            strict=True,
        )
    )
    morphology_loss = torch.stack(morphology_feature_losses).mean()

    lemma_rule_loss = calculate_categorical_distillation_loss(
        student_logits=student_logits.lemma_rule_logits,
        teacher_logits=teacher_logits.lemma_rule_logits,
        token_mask=token_mask & lemma_rule_mask,
        temperature=temperature,
    )

    total_loss = upos_loss + morphology_loss + lemma_rule_loss

    return TokenTaskLosses(
        upos_loss=upos_loss,
        morphology_loss=morphology_loss,
        lemma_rule_loss=lemma_rule_loss,
        total_loss=total_loss,
    )


def combine_token_task_losses(
    *,
    supervised_losses: TokenTaskLosses,
    distillation_losses: TokenTaskLosses,
    distillation_weight: float,
) -> CombinedTokenTaskLosses:
    if not math.isfinite(distillation_weight) or distillation_weight < 0.0:
        raise ValueError("Distillation weight must be finite and non-negative.")

    total_loss = (
        supervised_losses.total_loss
        + distillation_weight * distillation_losses.total_loss
    )

    return CombinedTokenTaskLosses(
        supervised_losses=supervised_losses,
        distillation_losses=distillation_losses,
        total_loss=total_loss,
    )
