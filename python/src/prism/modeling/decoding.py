import torch
from torch.nn import functional as F

from prism.modeling.outputs import (
    TokenTaskLogits,
    TokenTaskPredictionBatch,
)
from prism.schema import MorphologyFeatureSchema, MorphologySchema


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
