import math
from collections.abc import Callable
from dataclasses import dataclass

from prism.training.config import CheckpointSelectionMetric
from prism.training.epochs import (
    DistilledEpochMetrics,
    MixedEpochMetrics,
    SupervisedEpochMetrics,
    SupervisedEvaluationMetrics,
)

type TrainingEpochMetrics = (
    SupervisedEpochMetrics | DistilledEpochMetrics | MixedEpochMetrics
)


def _selection_score(
    metric: CheckpointSelectionMetric,
    development_metrics: SupervisedEvaluationMetrics,
) -> float:
    """Return a higher-is-better score for the configured selection metric."""

    if metric is CheckpointSelectionMetric.DEVELOPMENT_LOSS:
        return -development_metrics.losses.total_loss

    if development_metrics.morphology_bundle_exact_accuracy is None:
        raise ValueError(
            "Development-task-accuracy selection requires the exact "
            "morphology-bundle accuracy from the development evaluation."
        )
    accuracies = [
        development_metrics.upos_accuracy,
        development_metrics.morphology_bundle_exact_accuracy,
    ]
    if development_metrics.lemma_rule_accuracy is not None:
        accuracies.append(development_metrics.lemma_rule_accuracy)
    return sum(accuracies) / len(accuracies)


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedTrainingEpochResult:
    epoch_index: int
    training_metrics: TrainingEpochMetrics
    development_metrics: SupervisedEvaluationMetrics


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedTrainingRunResult:
    epoch_results: tuple[
        SupervisedTrainingEpochResult,
        ...,
    ]
    best_epoch_index: int
    stopped_early: bool = False
    secondary_best_epoch_index: int | None = None


def run_supervised_training_epochs(
    *,
    epoch_count: int,
    train_epoch: Callable[
        [int],
        TrainingEpochMetrics,
    ],
    evaluate_epoch: Callable[
        [int],
        SupervisedEvaluationMetrics,
    ],
    on_new_best: Callable[
        [SupervisedTrainingEpochResult],
        None,
    ],
    early_stopping_patience: int | None = None,
    checkpoint_selection_metric: CheckpointSelectionMetric = (
        CheckpointSelectionMetric.DEVELOPMENT_LOSS
    ),
    secondary_selection_metric: CheckpointSelectionMetric | None = None,
    on_new_secondary_best: (
        Callable[[SupervisedTrainingEpochResult], None] | None
    ) = None,
) -> SupervisedTrainingRunResult:
    """Run all epochs and track the best checkpoint per selection metric.

    The primary ``checkpoint_selection_metric`` drives ``best_epoch_index``,
    ``on_new_best``, and early stopping. The optional secondary metric is
    purely observational: it fires ``on_new_secondary_best`` whenever its own
    score improves, so one run can retain both selection candidates for a
    controlled selection-policy ablation. Early stopping still follows the
    primary metric only, so a secondary best beyond the stopping point is
    never observed; that boundary is inherent to the shared run.
    """

    if epoch_count <= 0:
        raise ValueError("Epoch count must be positive.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("Early-stopping patience must be positive.")
    if (secondary_selection_metric is None) != (on_new_secondary_best is None):
        raise ValueError(
            "Secondary selection requires both the metric and its callback."
        )
    if secondary_selection_metric is checkpoint_selection_metric:
        raise ValueError(
            "Secondary selection metric must differ from the primary metric."
        )

    epoch_results: list[SupervisedTrainingEpochResult] = []
    best_epoch_index: int | None = None
    best_selection_score = -math.inf
    secondary_best_epoch_index: int | None = None
    best_secondary_score = -math.inf
    epochs_without_improvement = 0
    stopped_early = False

    for epoch_index in range(epoch_count):
        training_metrics = train_epoch(epoch_index)
        development_metrics = evaluate_epoch(epoch_index)

        if not math.isfinite(development_metrics.losses.total_loss):
            raise ValueError("Development loss must be finite.")
        selection_score = _selection_score(
            checkpoint_selection_metric,
            development_metrics,
        )
        if not math.isfinite(selection_score):
            raise ValueError("Checkpoint selection score must be finite.")

        epoch_result = SupervisedTrainingEpochResult(
            epoch_index=epoch_index,
            training_metrics=training_metrics,
            development_metrics=development_metrics,
        )
        epoch_results.append(epoch_result)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch_index = epoch_index
            on_new_best(epoch_result)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if secondary_selection_metric is not None:
            secondary_score = _selection_score(
                secondary_selection_metric,
                development_metrics,
            )
            if not math.isfinite(secondary_score):
                raise ValueError("Secondary selection score must be finite.")
            if secondary_score > best_secondary_score:
                best_secondary_score = secondary_score
                secondary_best_epoch_index = epoch_index
                assert on_new_secondary_best is not None
                on_new_secondary_best(epoch_result)

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = epoch_index + 1 < epoch_count
            break

    if best_epoch_index is None:
        raise ValueError("Training run did not produce a best epoch.")

    return SupervisedTrainingRunResult(
        epoch_results=tuple(epoch_results),
        best_epoch_index=best_epoch_index,
        stopped_early=stopped_early,
        secondary_best_epoch_index=secondary_best_epoch_index,
    )
