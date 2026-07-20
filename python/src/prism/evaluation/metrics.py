from dataclasses import dataclass

import torch
from torch import Tensor

from prism.data import TokenTaskTargetBatch
from prism.modeling.outputs import TokenTaskPredictionBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskEvaluationCounts:
    token_count: Tensor
    upos_correct_count: Tensor
    morphology_correct_counts: tuple[Tensor, ...]
    morphology_annotated_counts: tuple[Tensor, ...]
    morphology_annotated_correct_counts: tuple[Tensor, ...]
    lemma_target_count: Tensor
    lemma_rule_correct_count: Tensor
    morphology_true_positive_counts: tuple[Tensor, ...]
    morphology_false_positive_counts: tuple[Tensor, ...]
    morphology_false_negative_counts: tuple[Tensor, ...]


def count_token_task_predictions(
    *,
    predictions: TokenTaskPredictionBatch,
    targets: TokenTaskTargetBatch,
) -> TokenTaskEvaluationCounts:
    if len(predictions.morphology_predictions) != len(targets.morphology_targets):
        raise ValueError("Predictions and target morphology features must match.")

    if not torch.equal(predictions.token_mask, targets.token_mask):
        raise ValueError("Prediction and target token masks must match.")

    token_mask = targets.token_mask
    lemma_mask = token_mask & targets.lemma_rule_mask

    upos_correct_count = ((predictions.upos_ids == targets.upos_ids) & token_mask).sum()

    morphology_correct_counts: list[Tensor] = []
    morphology_annotated_counts: list[Tensor] = []
    morphology_annotated_correct_counts: list[Tensor] = []
    morphology_true_positive_counts: list[Tensor] = []
    morphology_false_positive_counts: list[Tensor] = []
    morphology_false_negative_counts: list[Tensor] = []

    for feature_predictions, feature_targets in zip(
        predictions.morphology_predictions,
        targets.morphology_targets,
        strict=True,
    ):
        if feature_predictions.shape != feature_targets.shape:
            raise ValueError("Morphology prediction and target shapes must match.")

        correct = (feature_predictions == feature_targets).all(dim=-1)

        # Index 0 is always <NONE> according to the schema.
        annotated = ~feature_targets[..., 0]
        annotated_mask = token_mask & annotated
        label_mask = token_mask.unsqueeze(-1)

        true_positive = feature_predictions & feature_targets & label_mask
        false_positive = feature_predictions & ~feature_targets & label_mask
        false_negative = ~feature_predictions & feature_targets & label_mask

        morphology_true_positive_counts.append(true_positive.sum(dim=(0, 1)))
        morphology_false_positive_counts.append(false_positive.sum(dim=(0, 1)))
        morphology_false_negative_counts.append(false_negative.sum(dim=(0, 1)))

        morphology_correct_counts.append((correct & token_mask).sum())
        morphology_annotated_counts.append(annotated_mask.sum())
        morphology_annotated_correct_counts.append((correct & annotated_mask).sum())

    lemma_rule_correct_count = (
        (predictions.lemma_rule_ids == targets.lemma_rule_ids) & lemma_mask
    ).sum()

    return TokenTaskEvaluationCounts(
        token_count=token_mask.sum(),
        upos_correct_count=upos_correct_count,
        morphology_correct_counts=tuple(morphology_correct_counts),
        morphology_annotated_counts=tuple(morphology_annotated_counts),
        morphology_annotated_correct_counts=tuple(morphology_annotated_correct_counts),
        lemma_target_count=lemma_mask.sum(),
        lemma_rule_correct_count=lemma_rule_correct_count,
        morphology_true_positive_counts=tuple(morphology_true_positive_counts),
        morphology_false_positive_counts=tuple(morphology_false_positive_counts),
        morphology_false_negative_counts=tuple(morphology_false_negative_counts),
    )
