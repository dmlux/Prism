from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenTaskLogits


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskLossWeights:
    morphology_positive_weights: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.morphology_positive_weights:
            raise ValueError("Loss weights must contain morphology features.")

        for weights in self.morphology_positive_weights:
            if weights.ndim != 1:
                raise ValueError("Morphology positive weights must have one dimension.")
            if not weights.is_floating_point():
                raise ValueError("Morphology positive weights must be floating point.")
            if not torch.isfinite(weights).all():
                raise ValueError("Morphology positive weights must be finite.")
            if torch.any(weights <= 0):
                raise ValueError("Morphology positive weights must be positive.")

    def to(
        self,
        device: torch.device,
    ) -> "TokenTaskLossWeights":
        return TokenTaskLossWeights(
            morphology_positive_weights=tuple(
                weights.to(device=device)
                for weights in self.morphology_positive_weights
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskLosses:
    upos_loss: Tensor
    morphology_loss: Tensor
    lemma_rule_loss: Tensor
    total_loss: Tensor


def _masked_mean(
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    selected_values = values.masked_select(mask)

    if selected_values.numel() == 0:
        return values.sum() * 0.0

    return selected_values.mean()


def compute_token_task_loss(
    *,
    logits: TokenTaskLogits,
    targets: TokenTaskTargetBatch,
    loss_weights: TokenTaskLossWeights | None = None,
) -> TokenTaskLosses:
    if logits.upos_logits.shape[:2] != targets.upos_ids.shape:
        raise ValueError("Logits and targets must share batch and token dimensions.")
    if logits.morphology_feature_count != targets.morphology_feature_count:
        raise ValueError(
            "Morphology logits and targets must contain the same features."
        )

    upos_per_token = functional.cross_entropy(
        logits.upos_logits.transpose(1, 2),
        targets.upos_ids,
        reduction="none",
    )
    upos_loss = _masked_mean(
        upos_per_token,
        targets.token_mask,
    )

    morphology_feature_losses: list[Tensor] = []

    if loss_weights is None:
        morphology_positive_weights: tuple[Tensor | None, ...] = (
            None,
        ) * logits.morphology_feature_count
    else:
        if (
            len(loss_weights.morphology_positive_weights)
            != logits.morphology_feature_count
        ):
            raise ValueError("Morphology loss weights must match the feature count.")

        morphology_positive_weights = loss_weights.morphology_positive_weights

    for feature_logits, feature_targets, positive_weights in zip(
        logits.morphology_logits,
        targets.morphology_targets,
        morphology_positive_weights,
        strict=True,
    ):
        if feature_logits.shape != feature_targets.shape:
            raise ValueError(
                "Morphology logits must match morphology target dimensions."
            )
        if (
            positive_weights is not None
            and positive_weights.shape[0] != feature_logits.shape[-1]
        ):
            raise ValueError("Morphology positive weights must match the label count.")

        feature_per_label = functional.binary_cross_entropy_with_logits(
            feature_logits,
            feature_targets.to(feature_logits.dtype),
            pos_weight=positive_weights,
            reduction="none",
        )
        feature_per_token = feature_per_label.mean(dim=-1)
        morphology_feature_losses.append(
            _masked_mean(
                feature_per_token,
                targets.token_mask,
            )
        )

    morphology_loss = torch.stack(morphology_feature_losses).mean()

    lemma_per_token = functional.cross_entropy(
        logits.lemma_rule_logits.transpose(1, 2),
        targets.lemma_rule_ids,
        reduction="none",
    )
    lemma_rule_loss = _masked_mean(
        lemma_per_token,
        targets.token_mask & targets.lemma_rule_mask,
    )

    total_loss = upos_loss + morphology_loss + lemma_rule_loss

    return TokenTaskLosses(
        upos_loss=upos_loss,
        morphology_loss=morphology_loss,
        lemma_rule_loss=lemma_rule_loss,
        total_loss=total_loss,
    )
