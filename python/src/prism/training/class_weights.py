import math
from collections.abc import Sequence

import torch
from torch import Tensor

from prism.data import TokenTargets
from prism.schema import MorphologySchema
from prism.training.config import SupervisedTrainingConfig
from prism.training.losses import TokenTaskLossWeights


def calculate_binary_positive_weights(
    *,
    targets: Tensor,
    token_mask: Tensor,
    maximum_weight: float | None = None,
) -> Tensor:
    if targets.ndim != 3:
        raise ValueError("Targets must have batch, token, and label dimensions.")

    if token_mask.shape != targets.shape[:2]:
        raise ValueError("Token mask must match the batch and token dimensions.")

    if targets.dtype != torch.bool:
        raise ValueError("Targets must contain boolean values.")

    if maximum_weight is not None and (
        not math.isfinite(maximum_weight) or maximum_weight < 1.0
    ):
        raise ValueError("Maximum positive weight must be finite and at least one.")

    valid_targets = targets[token_mask]
    label_count = targets.shape[-1]

    if valid_targets.shape[0] == 0:
        return torch.ones(
            label_count,
            dtype=torch.float32,
            device=targets.device,
        )

    positive_counts = valid_targets.sum(dim=0).to(torch.float32)
    negative_counts = valid_targets.shape[0] - positive_counts

    positive_ratio = negative_counts / positive_counts.clamp_min(1.0)

    weights = torch.sqrt(positive_ratio.clamp_min(1.0))

    if maximum_weight is not None:
        weights = weights.clamp_max(maximum_weight)

    return torch.where(
        positive_counts > 0,
        weights,
        torch.ones_like(weights),
    )


def calculate_morphology_weights(
    *,
    targets: Sequence[TokenTargets],
    morphology_schema: MorphologySchema,
    maximum_weight: float,
) -> tuple[Tensor, ...]:
    if not targets:
        raise ValueError("Morphology weight calculation requires token targets.")

    morphology_feature_count = len(targets[0].morphology)
    label_counts = tuple(len(labels) for labels in targets[0].morphology)

    if morphology_feature_count != len(morphology_schema.features):
        raise ValueError("Morphology targets must match the morphology schema.")

    for target in targets:
        if len(target.morphology) != morphology_feature_count:
            raise ValueError("All targets must contain the same morphology features.")

        if tuple(len(labels) for labels in target.morphology) != label_counts:
            raise ValueError("Morphology label counts must remain consistent.")

    token_mask = torch.ones(
        (1, len(targets)),
        dtype=torch.bool,
    )

    weights: list[Tensor] = []

    for feature_index, feature_schema in enumerate(morphology_schema.features):
        feature_targets = torch.tensor(
            [[target.morphology[feature_index] for target in targets]],
            dtype=torch.bool,
        )
        if feature_targets.shape[-1] != len(feature_schema.labels):
            raise ValueError(
                "Morphology target labels must match the morphology schema."
            )
        if feature_schema.allows_multiple_values:
            feature_targets = feature_targets[..., 1:]

        weights.append(
            calculate_binary_positive_weights(
                targets=feature_targets,
                token_mask=token_mask,
                maximum_weight=maximum_weight,
            )
        )

    return tuple(weights)


def build_token_task_loss_weights(
    *,
    targets: Sequence[TokenTargets],
    morphology_schema: MorphologySchema,
    config: SupervisedTrainingConfig,
) -> TokenTaskLossWeights | None:
    maximum_weight = config.morphology_weight_cap

    if maximum_weight is None:
        return None

    return TokenTaskLossWeights(
        morphology_weights=(
            calculate_morphology_weights(
                targets=targets,
                morphology_schema=morphology_schema,
                maximum_weight=maximum_weight,
            )
        ),
    )
