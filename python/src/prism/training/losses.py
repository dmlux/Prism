from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenTaskLogits


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

    for feature_logits, feature_targets in zip(
        logits.morphology_logits,
        targets.morphology_targets,
        strict=True,
    ):
        if feature_logits.shape != feature_targets.shape:
            raise ValueError(
                "Morphology logits must match morphology target dimensions."
            )

        feature_per_label = functional.binary_cross_entropy_with_logits(
            feature_logits,
            feature_targets.to(feature_logits.dtype),
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
