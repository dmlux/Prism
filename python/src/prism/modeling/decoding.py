import torch
from torch.nn import functional as F

from prism.modeling.outputs import (
    TokenTaskLogits,
    TokenTaskPredictionBatch,
)
from prism.schema import MorphologySchema


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

        if feature_logits.shape[-1] != label_count:
            raise ValueError("Morphology logit count must match schema labels.")

        if feature_schema.allows_multiple_values:
            value_predictions = feature_logits[..., 1:] > 0.0
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
