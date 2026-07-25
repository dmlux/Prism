import pytest

from prism.training.config import CheckpointSelectionMetric
from prism.training.epochs import (
    SupervisedEpochMetrics,
    SupervisedEvaluationMetrics,
)
from prism.training.runner import (
    run_supervised_training_epochs,
)


def _loss_metrics(total_loss: float) -> SupervisedEpochMetrics:
    return SupervisedEpochMetrics(
        batch_count=1,
        token_count=10,
        lemma_target_count=10,
        upos_loss=total_loss,
        morphology_loss=0.0,
        lemma_rule_loss=0.0,
    )


def _evaluation_metrics(
    total_loss: float,
    *,
    upos_accuracy: float = 0.5,
    lemma_rule_accuracy: float | None = 0.5,
    morphology_bundle_exact_accuracy: float | None = 0.5,
) -> SupervisedEvaluationMetrics:
    return SupervisedEvaluationMetrics(
        losses=_loss_metrics(total_loss),
        upos_accuracy=upos_accuracy,
        morphology_bundle_exact_accuracy=morphology_bundle_exact_accuracy,
        morphology_accuracies=(0.5,),
        morphology_annotated_accuracies=(0.5,),
        lemma_rule_accuracy=lemma_rule_accuracy,
        morphology_true_positive_counts=((1, 0),),
        morphology_false_positive_counts=((0, 0),),
        morphology_false_negative_counts=((0, 1),),
        morphology_average_precisions=((0.5, 0.5),),
    )


def test_training_runner_selects_lowest_development_loss() -> None:
    development_losses = (0.8, 0.4, 0.6)
    trained_epoch_indices: list[int] = []
    evaluated_epoch_indices: list[int] = []
    improved_epoch_indices: list[int] = []

    def train_epoch(
        epoch_index: int,
    ) -> SupervisedEpochMetrics:
        trained_epoch_indices.append(epoch_index)
        return _loss_metrics(1.0 - epoch_index * 0.1)

    def evaluate_epoch(
        epoch_index: int,
    ) -> SupervisedEvaluationMetrics:
        evaluated_epoch_indices.append(epoch_index)
        return _evaluation_metrics(development_losses[epoch_index])

    result = run_supervised_training_epochs(
        epoch_count=3,
        train_epoch=train_epoch,
        evaluate_epoch=evaluate_epoch,
        on_new_best=lambda epoch: improved_epoch_indices.append(epoch.epoch_index),
    )

    assert trained_epoch_indices == [0, 1, 2]
    assert evaluated_epoch_indices == [0, 1, 2]
    assert improved_epoch_indices == [0, 1]

    assert len(result.epoch_results) == 3
    assert result.best_epoch_index == 1
    assert result.epoch_results[1].development_metrics.losses.total_loss == 0.4


def test_training_runner_selects_highest_task_accuracy() -> None:
    # The loss keeps worsening while the discrete decisions keep improving,
    # mirroring the observed overconfident teacher learning curves.
    development_losses = (0.11, 0.13, 0.16)
    task_accuracies = (
        (0.989, 0.988, 0.905),
        (0.991, 0.990, 0.915),
        (0.991, 0.991, 0.930),
    )
    improved_epoch_indices: list[int] = []

    def evaluate_epoch(epoch_index: int) -> SupervisedEvaluationMetrics:
        upos, lemma, bundle = task_accuracies[epoch_index]
        return _evaluation_metrics(
            development_losses[epoch_index],
            upos_accuracy=upos,
            lemma_rule_accuracy=lemma,
            morphology_bundle_exact_accuracy=bundle,
        )

    result = run_supervised_training_epochs(
        epoch_count=3,
        train_epoch=lambda _: _loss_metrics(1.0),
        evaluate_epoch=evaluate_epoch,
        on_new_best=lambda epoch: improved_epoch_indices.append(epoch.epoch_index),
        checkpoint_selection_metric=(
            CheckpointSelectionMetric.DEVELOPMENT_TASK_ACCURACY
        ),
    )

    assert improved_epoch_indices == [0, 1, 2]
    assert result.best_epoch_index == 2


def test_task_accuracy_selection_requires_bundle_accuracy() -> None:
    with pytest.raises(ValueError, match="exact"):
        run_supervised_training_epochs(
            epoch_count=1,
            train_epoch=lambda _: _loss_metrics(1.0),
            evaluate_epoch=lambda index: _evaluation_metrics(
                0.5,
                morphology_bundle_exact_accuracy=None,
            ),
            on_new_best=lambda _: None,
            checkpoint_selection_metric=(
                CheckpointSelectionMetric.DEVELOPMENT_TASK_ACCURACY
            ),
        )


def test_training_runner_tracks_secondary_best_independently() -> None:
    # Loss keeps improving through epoch 2 while the discrete decisions peak
    # at epoch 1: the primary metric selects epoch 2, the secondary epoch 1.
    development_losses = (0.30, 0.20, 0.10)
    bundle_accuracies = (0.90, 0.95, 0.93)
    primary_epochs: list[int] = []
    secondary_epochs: list[int] = []

    result = run_supervised_training_epochs(
        epoch_count=3,
        train_epoch=lambda _: _loss_metrics(1.0),
        evaluate_epoch=lambda index: _evaluation_metrics(
            development_losses[index],
            morphology_bundle_exact_accuracy=bundle_accuracies[index],
        ),
        on_new_best=lambda epoch: primary_epochs.append(epoch.epoch_index),
        checkpoint_selection_metric=CheckpointSelectionMetric.DEVELOPMENT_LOSS,
        secondary_selection_metric=(
            CheckpointSelectionMetric.DEVELOPMENT_TASK_ACCURACY
        ),
        on_new_secondary_best=(
            lambda epoch: secondary_epochs.append(epoch.epoch_index)
        ),
    )

    assert primary_epochs == [0, 1, 2]
    assert secondary_epochs == [0, 1]
    assert result.best_epoch_index == 2
    assert result.secondary_best_epoch_index == 1


def test_training_runner_rejects_inconsistent_secondary_selection() -> None:
    with pytest.raises(ValueError, match="must differ"):
        run_supervised_training_epochs(
            epoch_count=1,
            train_epoch=lambda _: _loss_metrics(1.0),
            evaluate_epoch=lambda index: _evaluation_metrics(0.5),
            on_new_best=lambda _: None,
            secondary_selection_metric=(
                CheckpointSelectionMetric.DEVELOPMENT_LOSS
            ),
            on_new_secondary_best=lambda _: None,
        )

    with pytest.raises(ValueError, match="metric and its callback"):
        run_supervised_training_epochs(
            epoch_count=1,
            train_epoch=lambda _: _loss_metrics(1.0),
            evaluate_epoch=lambda index: _evaluation_metrics(0.5),
            on_new_best=lambda _: None,
            secondary_selection_metric=(
                CheckpointSelectionMetric.DEVELOPMENT_TASK_ACCURACY
            ),
        )


def test_training_runner_waits_four_epochs_before_stopping() -> None:
    development_losses = (0.8, 0.7, 0.6, 0.61, 0.62, 0.63, 0.59, 0.60)

    result = run_supervised_training_epochs(
        epoch_count=len(development_losses),
        train_epoch=lambda _: _loss_metrics(1.0),
        evaluate_epoch=lambda index: _evaluation_metrics(development_losses[index]),
        on_new_best=lambda _: None,
        early_stopping_patience=4,
    )

    assert not result.stopped_early
    assert len(result.epoch_results) == len(development_losses)
    assert result.best_epoch_index == 6


def test_training_runner_stops_after_configured_patience() -> None:
    development_losses = (0.8, 0.7, 0.71, 0.72, 0.73, 0.74, 0.6)

    result = run_supervised_training_epochs(
        epoch_count=len(development_losses),
        train_epoch=lambda _: _loss_metrics(1.0),
        evaluate_epoch=lambda index: _evaluation_metrics(development_losses[index]),
        on_new_best=lambda _: None,
        early_stopping_patience=4,
    )

    assert result.stopped_early
    assert len(result.epoch_results) == 6
    assert result.best_epoch_index == 1
