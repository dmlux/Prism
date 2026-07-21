from dataclasses import dataclass, field

import torch
from torch import Tensor

from prism.data import TokenTaskTargetBatch
from prism.evaluation.classification import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from prism.evaluation.ranking import calculate_average_precision
from prism.modeling.decoding import morphology_label_scores
from prism.modeling.outputs import (
    TokenTaskLogits,
    TokenTaskPredictionBatch,
)
from prism.schema import MorphologySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskEvaluationCounts:
    token_count: Tensor
    upos_correct_count: Tensor
    morphology_correct_counts: tuple[Tensor, ...]
    morphology_annotated_counts: tuple[Tensor, ...]
    morphology_annotated_correct_counts: tuple[Tensor, ...]
    lemma_target_count: Tensor
    lemma_annotation_count: Tensor
    lemma_rule_correct_count: Tensor
    morphology_true_positive_counts: tuple[Tensor, ...]
    morphology_false_positive_counts: tuple[Tensor, ...]
    morphology_false_negative_counts: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskEvaluationMetrics:
    token_count: int
    lemma_target_count: int
    lemma_annotation_count: int
    upos_accuracy: float
    morphology_accuracies: tuple[float, ...]
    morphology_annotated_accuracies: tuple[float | None, ...]
    lemma_rule_accuracy: float | None
    lemma_rule_coverage: float | None
    lemma_end_to_end_accuracy: float | None
    morphology_true_positive_counts: tuple[tuple[int, ...], ...]
    morphology_false_positive_counts: tuple[tuple[int, ...], ...]
    morphology_false_negative_counts: tuple[tuple[int, ...], ...]
    morphology_average_precisions: tuple[tuple[float | None, ...], ...]

    def morphology_micro_metrics(self) -> ClassificationMetrics:
        true_positive_count = sum(
            sum(feature_counts[1:])
            for feature_counts in self.morphology_true_positive_counts
        )
        false_positive_count = sum(
            sum(feature_counts[1:])
            for feature_counts in self.morphology_false_positive_counts
        )
        false_negative_count = sum(
            sum(feature_counts[1:])
            for feature_counts in self.morphology_false_negative_counts
        )

        return calculate_classification_metrics(
            true_positive_count=true_positive_count,
            false_positive_count=false_positive_count,
            false_negative_count=false_negative_count,
        )


def count_token_task_predictions(
    *,
    predictions: TokenTaskPredictionBatch,
    targets: TokenTaskTargetBatch,
    evaluation_mask: Tensor | None = None,
) -> TokenTaskEvaluationCounts:
    if len(predictions.morphology_predictions) != len(targets.morphology_targets):
        raise ValueError("Predictions and target morphology features must match.")

    if not torch.equal(predictions.token_mask, targets.token_mask):
        raise ValueError("Prediction and target token masks must match.")

    if evaluation_mask is None:
        token_mask = targets.token_mask
    else:
        if evaluation_mask.shape != targets.token_mask.shape:
            raise ValueError("Evaluation mask must match target token dimensions.")
        if evaluation_mask.dtype != torch.bool:
            raise ValueError("Evaluation mask must use torch.bool.")
        if evaluation_mask.device != targets.token_mask.device:
            raise ValueError("Evaluation mask and targets must use the same device.")
        if (evaluation_mask & ~targets.token_mask).any().item():
            raise ValueError("Evaluation mask must not select padding tokens.")

        token_mask = evaluation_mask
    if targets.lemma_annotation_mask is None:
        raise RuntimeError("Lemma annotation mask must be resolved.")

    lemma_mask = token_mask & targets.lemma_rule_mask
    lemma_annotation_mask = token_mask & targets.lemma_annotation_mask

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
        lemma_annotation_count=lemma_annotation_mask.sum(),
        lemma_rule_correct_count=lemma_rule_correct_count,
        morphology_true_positive_counts=tuple(morphology_true_positive_counts),
        morphology_false_positive_counts=tuple(morphology_false_positive_counts),
        morphology_false_negative_counts=tuple(morphology_false_negative_counts),
    )


@dataclass(slots=True, kw_only=True)
class TokenTaskEvaluationAccumulator:
    device: torch.device
    morphology_schema: MorphologySchema
    token_count: Tensor = field(init=False)
    upos_correct_count: Tensor = field(init=False)
    lemma_target_count: Tensor = field(init=False)
    lemma_annotation_count: Tensor = field(init=False)
    lemma_rule_correct_count: Tensor = field(init=False)
    morphology_correct_counts: tuple[Tensor, ...] = field(init=False)
    morphology_annotated_counts: tuple[Tensor, ...] = field(init=False)
    morphology_annotated_correct_counts: tuple[Tensor, ...] = field(init=False)
    morphology_true_positive_counts: tuple[Tensor, ...] = field(init=False)
    morphology_false_positive_counts: tuple[Tensor, ...] = field(init=False)
    morphology_false_negative_counts: tuple[Tensor, ...] = field(init=False)
    morphology_score_batches: tuple[list[Tensor], ...] = field(init=False)
    morphology_target_batches: tuple[list[Tensor], ...] = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.upos_correct_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.lemma_target_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.lemma_annotation_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.lemma_rule_correct_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.morphology_correct_counts = self._feature_scalars()
        self.morphology_annotated_counts = self._feature_scalars()
        self.morphology_annotated_correct_counts = self._feature_scalars()
        self.morphology_true_positive_counts = self._feature_label_vectors()
        self.morphology_false_positive_counts = self._feature_label_vectors()
        self.morphology_false_negative_counts = self._feature_label_vectors()
        self.morphology_score_batches = tuple(
            [] for _ in self.morphology_schema.features
        )
        self.morphology_target_batches = tuple(
            [] for _ in self.morphology_schema.features
        )

    def _feature_scalars(self) -> tuple[Tensor, ...]:
        return tuple(
            torch.zeros((), dtype=torch.long, device=self.device)
            for _ in self.morphology_schema.features
        )

    def _feature_label_vectors(self) -> tuple[Tensor, ...]:
        return tuple(
            torch.zeros(
                len(feature.labels),
                dtype=torch.long,
                device=self.device,
            )
            for feature in self.morphology_schema.features
        )

    def add(
        self,
        *,
        logits: TokenTaskLogits,
        predictions: TokenTaskPredictionBatch,
        targets: TokenTaskTargetBatch,
        evaluation_mask: Tensor,
    ) -> None:
        counts = count_token_task_predictions(
            predictions=predictions,
            targets=targets,
            evaluation_mask=evaluation_mask,
        )

        self.token_count += counts.token_count
        self.upos_correct_count += counts.upos_correct_count
        self.lemma_target_count += counts.lemma_target_count
        self.lemma_annotation_count += counts.lemma_annotation_count
        self.lemma_rule_correct_count += counts.lemma_rule_correct_count

        for totals, currents in (
            (self.morphology_correct_counts, counts.morphology_correct_counts),
            (self.morphology_annotated_counts, counts.morphology_annotated_counts),
            (
                self.morphology_annotated_correct_counts,
                counts.morphology_annotated_correct_counts,
            ),
            (
                self.morphology_true_positive_counts,
                counts.morphology_true_positive_counts,
            ),
            (
                self.morphology_false_positive_counts,
                counts.morphology_false_positive_counts,
            ),
            (
                self.morphology_false_negative_counts,
                counts.morphology_false_negative_counts,
            ),
        ):
            for total, current in zip(totals, currents, strict=True):
                total.add_(current)

        for (
            score_batches,
            target_batches,
            feature_logits,
            feature_targets,
            feature,
        ) in zip(
            self.morphology_score_batches,
            self.morphology_target_batches,
            logits.morphology_logits,
            targets.morphology_targets,
            self.morphology_schema.features,
            strict=True,
        ):
            score_batches.append(
                morphology_label_scores(
                    feature_logits=feature_logits,
                    feature_schema=feature,
                )[evaluation_mask]
                .detach()
                .cpu()
            )
            target_batches.append(feature_targets[evaluation_mask].detach().cpu())

    def finish(self, *, empty_slice_message: str) -> TokenTaskEvaluationMetrics:
        token_count = int(self.token_count.item())
        if token_count == 0:
            raise ValueError(empty_slice_message)

        lemma_target_count = int(self.lemma_target_count.item())
        lemma_annotation_count = int(self.lemma_annotation_count.item())
        morphology_average_precisions = tuple(
            self._calculate_feature_average_precisions(
                score_batches=score_batches,
                target_batches=target_batches,
            )
            for score_batches, target_batches in zip(
                self.morphology_score_batches,
                self.morphology_target_batches,
                strict=True,
            )
        )

        return TokenTaskEvaluationMetrics(
            token_count=token_count,
            lemma_target_count=lemma_target_count,
            lemma_annotation_count=lemma_annotation_count,
            upos_accuracy=(self.upos_correct_count / self.token_count).item(),
            morphology_accuracies=tuple(
                (correct_count / self.token_count).item()
                for correct_count in self.morphology_correct_counts
            ),
            morphology_annotated_accuracies=tuple(
                None
                if annotated_count.item() == 0
                else (correct_count / annotated_count).item()
                for correct_count, annotated_count in zip(
                    self.morphology_annotated_correct_counts,
                    self.morphology_annotated_counts,
                    strict=True,
                )
            ),
            lemma_rule_accuracy=(
                None
                if lemma_target_count == 0
                else (self.lemma_rule_correct_count / self.lemma_target_count).item()
            ),
            lemma_rule_coverage=(
                None
                if lemma_annotation_count == 0
                else lemma_target_count / lemma_annotation_count
            ),
            lemma_end_to_end_accuracy=(
                None
                if lemma_annotation_count == 0
                else (
                    self.lemma_rule_correct_count / self.lemma_annotation_count
                ).item()
            ),
            morphology_true_positive_counts=self._integer_counts(
                self.morphology_true_positive_counts
            ),
            morphology_false_positive_counts=self._integer_counts(
                self.morphology_false_positive_counts
            ),
            morphology_false_negative_counts=self._integer_counts(
                self.morphology_false_negative_counts
            ),
            morphology_average_precisions=morphology_average_precisions,
        )

    @staticmethod
    def _integer_counts(counts: tuple[Tensor, ...]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(value) for value in feature_counts.detach().cpu().tolist())
            for feature_counts in counts
        )

    @staticmethod
    def _calculate_feature_average_precisions(
        *,
        score_batches: list[Tensor],
        target_batches: list[Tensor],
    ) -> tuple[float | None, ...]:
        feature_scores = torch.cat(score_batches, dim=0)
        feature_targets = torch.cat(target_batches, dim=0)

        return tuple(
            calculate_average_precision(
                scores=feature_scores[:, label_index],
                targets=feature_targets[:, label_index],
            )
            for label_index in range(feature_scores.shape[-1])
        )
