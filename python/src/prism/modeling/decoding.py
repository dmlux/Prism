from dataclasses import dataclass

import torch
from torch.nn import functional as F

from prism.modeling.outputs import (
    TokenTaskLogits,
    TokenTaskPredictionBatch,
)
from prism.schema import MorphologyFeatureSchema, MorphologySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyLogitCorrection:
    """Undo a configurable share of morphology training-weight logit shifts."""

    strength: float
    weights: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                "Morphology logit-correction strength must be between zero and one."
            )
        if not self.weights:
            raise ValueError("Morphology logit correction requires feature weights.")
        for weights in self.weights:
            if weights.ndim != 1:
                raise ValueError(
                    "Morphology logit-correction weights must have one dimension."
                )
            if not weights.is_floating_point():
                raise ValueError(
                    "Morphology logit-correction weights must be floating point."
                )
            if not torch.isfinite(weights).all().item():
                raise ValueError("Morphology logit-correction weights must be finite.")
            if torch.any(weights <= 0.0).item():
                raise ValueError(
                    "Morphology logit-correction weights must be positive."
                )


def apply_morphology_logit_correction(
    *,
    logits: TokenTaskLogits,
    morphology_schema: MorphologySchema,
    correction: MorphologyLogitCorrection,
) -> TokenTaskLogits:
    """Remove the selected fraction of the weighted-loss prior from logits."""

    if len(logits.morphology_logits) != len(morphology_schema.features):
        raise ValueError("Morphology logits must match the morphology schema.")
    if len(correction.weights) != len(morphology_schema.features):
        raise ValueError(
            "Morphology logit-correction weights must match the feature count."
        )

    corrected_morphology_logits: list[torch.Tensor] = []
    for feature_logits, feature_schema, feature_weights in zip(
        logits.morphology_logits,
        morphology_schema.features,
        correction.weights,
        strict=True,
    ):
        if feature_logits.shape[-1] != feature_schema.logit_count:
            raise ValueError("Morphology logit count must match the feature schema.")
        if feature_weights.shape[0] != feature_schema.logit_count:
            raise ValueError(
                "Morphology logit-correction weights must match feature logits."
            )

        resolved_weights = feature_weights.to(
            device=feature_logits.device,
            dtype=feature_logits.dtype,
        )
        corrected_morphology_logits.append(
            feature_logits - correction.strength * resolved_weights.log()
        )

    return TokenTaskLogits(
        upos_logits=logits.upos_logits,
        morphology_logits=tuple(corrected_morphology_logits),
        lemma_rule_logits=logits.lemma_rule_logits,
    )


def morphology_label_scores(
    *,
    feature_logits: torch.Tensor,
    feature_schema: MorphologyFeatureSchema,
) -> torch.Tensor:
    expected_logit_count = feature_schema.logit_count
    if feature_logits.shape[-1] != expected_logit_count:
        raise ValueError("Morphology logit count must match the feature schema.")

    if not feature_schema.allows_multiple_values:
        return F.softmax(feature_logits, dim=-1)

    value_probabilities = torch.sigmoid(feature_logits)
    none_probability = (1.0 - value_probabilities).prod(
        dim=-1,
        keepdim=True,
    )
    return torch.cat((none_probability, value_probabilities), dim=-1)


def decode_token_task_logits(
    *,
    logits: TokenTaskLogits,
    token_mask: torch.Tensor,
    morphology_schema: MorphologySchema,
) -> TokenTaskPredictionBatch:
    if len(logits.morphology_logits) != len(morphology_schema.features):
        raise ValueError("Morphology logits must match the morphology schema.")

    morphology_predictions: list[torch.Tensor] = []

    for feature_logits, feature_schema in zip(
        logits.morphology_logits, morphology_schema.features, strict=True
    ):
        label_count = len(feature_schema.labels)

        if feature_logits.shape[-1] != feature_schema.logit_count:
            raise ValueError("Morphology logit count must match the feature schema.")

        if feature_schema.allows_multiple_values:
            value_predictions = feature_logits > 0.0
            none_predictions = ~value_predictions.any(
                dim=-1,
                keepdim=True,
            )

            predictions = torch.cat(
                (none_predictions, value_predictions),
                dim=-1,
            )
        else:
            prediction_ids = feature_logits.argmax(dim=-1)
            predictions = F.one_hot(
                prediction_ids,
                num_classes=label_count,
            ).to(dtype=torch.bool)

        morphology_predictions.append(predictions)

    return TokenTaskPredictionBatch(
        upos_ids=logits.upos_logits.argmax(dim=-1),
        morphology_predictions=tuple(morphology_predictions),
        lemma_rule_ids=logits.lemma_rule_logits.argmax(dim=-1),
        token_mask=token_mask,
    )
