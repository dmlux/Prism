import math
from collections.abc import Callable
from dataclasses import dataclass

from prism.training.epochs import (
    DistilledEpochMetrics,
    SupervisedEpochMetrics,
    SupervisedEvaluationMetrics,
)

type TrainingEpochMetrics = SupervisedEpochMetrics | DistilledEpochMetrics


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
) -> SupervisedTrainingRunResult:
    if epoch_count <= 0:
        raise ValueError("Epoch count must be positive.")

    epoch_results: list[SupervisedTrainingEpochResult] = []
    best_epoch_index: int | None = None
    best_development_loss = math.inf

    for epoch_index in range(epoch_count):
        training_metrics = train_epoch(epoch_index)
        development_metrics = evaluate_epoch(epoch_index)
        development_loss = development_metrics.losses.total_loss

        if not math.isfinite(development_loss):
            raise ValueError("Development loss must be finite.")

        epoch_result = SupervisedTrainingEpochResult(
            epoch_index=epoch_index,
            training_metrics=training_metrics,
            development_metrics=development_metrics,
        )
        epoch_results.append(epoch_result)

        if development_loss < best_development_loss:
            best_development_loss = development_loss
            best_epoch_index = epoch_index
            on_new_best(epoch_result)

    if best_epoch_index is None:
        raise ValueError("Training run did not produce a best epoch.")

    return SupervisedTrainingRunResult(
        epoch_results=tuple(epoch_results), best_epoch_index=best_epoch_index
    )
