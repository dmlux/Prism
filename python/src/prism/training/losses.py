from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenTaskLogits
from prism.schema import MorphologySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskLossWeights:
    morphology_weights: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.morphology_weights:
            raise ValueError("Loss weights must contain morphology features.")

        for weights in self.morphology_weights:
            if weights.ndim != 1:
                raise ValueError("Morphology weights must have one dimension.")
            if not weights.is_floating_point():
                raise ValueError("Morphology weights must be floating point.")
            if not torch.isfinite(weights).all():
                raise ValueError("Morphology weights must be finite.")
            if torch.any(weights <= 0):
                raise ValueError("Morphology weights must be positive.")

    def to(
        self,
        device: torch.device,
    ) -> "TokenTaskLossWeights":
        return TokenTaskLossWeights(
            morphology_weights=tuple(
                weights.to(device=device) for weights in self.morphology_weights
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
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
) -> TokenTaskLosses:
    if logits.upos_logits.shape[:2] != targets.upos_ids.shape:
        raise ValueError("Logits and targets must share batch and token dimensions.")
    if logits.morphology_feature_count != targets.morphology_feature_count:
        raise ValueError(
            "Morphology logits and targets must contain the same features."
        )
    if logits.morphology_feature_count != len(morphology_schema.features):
        raise ValueError("Morphology logits must match the morphology schema.")

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
        morphology_weights: tuple[Tensor | None, ...] = (
            None,
        ) * logits.morphology_feature_count
    else:
        if len(loss_weights.morphology_weights) != logits.morphology_feature_count:
            raise ValueError("Morphology loss weights must match the feature count.")

        morphology_weights = loss_weights.morphology_weights

    for feature_logits, feature_targets, feature_schema, feature_weights in zip(
        logits.morphology_logits,
        targets.morphology_targets,
        morphology_schema.features,
        morphology_weights,
        strict=True,
    ):
        expected_target_shape = (*feature_logits.shape[:2], len(feature_schema.labels))
        if feature_targets.shape != expected_target_shape:
            raise ValueError(
                "Morphology targets must match the feature's complete label space."
            )
        if (
            feature_weights is not None
            and feature_weights.shape[0] != feature_schema.logit_count
        ):
            raise ValueError("Morphology weights must match the feature logit count.")

        valid_targets = feature_targets[targets.token_mask]

        if feature_schema.allows_multiple_values:
            none_targets = valid_targets[..., 0]
            value_targets = valid_targets[..., 1:]
            if torch.any(none_targets != ~value_targets.any(dim=-1)):
                raise ValueError(
                    "Multi-label morphology targets must derive <NONE> from values."
                )

            feature_per_label = functional.binary_cross_entropy_with_logits(
                feature_logits,
                feature_targets[..., 1:].to(feature_logits.dtype),
                pos_weight=feature_weights,
                reduction="none",
            )
            feature_per_token = feature_per_label.mean(dim=-1)
        else:
            if torch.any(valid_targets.sum(dim=-1) != 1):
                raise ValueError(
                    "Categorical morphology targets must activate exactly one label."
                )

            target_ids = feature_targets.to(torch.long).argmax(dim=-1)
            feature_per_token = functional.cross_entropy(
                feature_logits.transpose(1, 2),
                target_ids,
                weight=feature_weights,
                reduction="none",
            )

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
