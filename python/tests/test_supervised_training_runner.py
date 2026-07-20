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
) -> SupervisedEvaluationMetrics:
    return SupervisedEvaluationMetrics(
        losses=_loss_metrics(total_loss),
        upos_accuracy=0.5,
        morphology_accuracies=(0.5,),
        morphology_annotated_accuracies=(0.5,),
        lemma_rule_accuracy=0.5,
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
