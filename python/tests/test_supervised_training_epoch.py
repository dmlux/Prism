import math

import torch
from torch import nn

from prism.data import TokenTaskTargetBatch
from prism.modeling import (
    MorphologyLogitCorrection,
    TokenizedBatch,
    TokenTaskLogits,
)
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.training import (
    SupervisedTokenTaskBatch,
    TokenTaskDistillationPolicy,
    TokenTaskLossWeights,
    build_linear_warmup_decay_scheduler,
    evaluate_supervised_token_task_epoch,
    train_distilled_token_task_epoch,
    train_supervised_token_task_epoch,
)
from prism.schema import (
    MorphologyFeatureSchema,
    MorphologySchema,
)


MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Feature",
            values=("Value",),
            allows_multiple_values=False,
        ),
    ),
)


class TinyEpochModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.upos = nn.Parameter(torch.zeros(2))
        self.morphology = nn.Parameter(torch.zeros(2))
        self.lemma_rules = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        batch: TokenizedBatch,
    ) -> TokenTaskLogits:
        dimensions = (
            batch.batch_size,
            batch.max_token_count,
            2,
        )

        return TokenTaskLogits(
            upos_logits=self.upos.expand(dimensions),
            morphology_logits=(self.morphology.expand(dimensions),),
            lemma_rule_logits=self.lemma_rules.expand(dimensions),
        )


def _training_batch(target_id: int) -> SupervisedTokenTaskBatch:
    return SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor(
                [[1, 2, 3]],
                dtype=torch.long,
            ),
            attention_mask=torch.tensor(
                [[True, True, True]],
                dtype=torch.bool,
            ),
            first_subword_indices=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            subword_end_indices=torch.tensor(
                [[2]],
                dtype=torch.long,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor(
                [[target_id]],
                dtype=torch.long,
            ),
            morphology_targets=(
                torch.tensor(
                    [[[target_id == 0, target_id == 1]]],
                    dtype=torch.bool,
                ),
            ),
            lemma_rule_ids=torch.tensor(
                [[target_id]],
                dtype=torch.long,
            ),
            lemma_rule_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
    )


def test_training_epoch_runs_all_batches_and_scheduler_steps() -> None:
    model = TinyEpochModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_step_count=2,
        warmup_ratio=0.0,
    )

    metrics = train_supervised_token_task_epoch(
        model=model,
        batches=(
            _training_batch(0),
            _training_batch(1),
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
        morphology_schema=MORPHOLOGY_SCHEMA,
    )

    assert metrics.batch_count == 2
    assert metrics.token_count == 2
    assert metrics.lemma_target_count == 2
    assert math.isfinite(metrics.upos_loss)
    assert math.isfinite(metrics.morphology_loss)
    assert math.isfinite(metrics.lemma_rule_loss)
    assert math.isfinite(metrics.total_loss)
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_evaluation_epoch_reports_task_accuracies() -> None:
    model = TinyEpochModel()
    parameters_before_evaluation = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    metrics = evaluate_supervised_token_task_epoch(
        model=model,
        batches=(
            _training_batch(0),
            _training_batch(1),
        ),
        device=torch.device("cpu"),
        morphology_schema=MORPHOLOGY_SCHEMA,
    )

    assert metrics.losses.batch_count == 2
    assert metrics.losses.token_count == 2
    assert metrics.upos_accuracy == 0.5
    assert metrics.morphology_accuracies == (0.5,)
    assert metrics.morphology_annotated_accuracies == (0.0,)
    assert metrics.lemma_rule_accuracy == 0.5

    assert metrics.morphology_true_positive_counts == ((1, 0),)
    assert metrics.morphology_false_positive_counts == ((1, 0),)
    assert metrics.morphology_false_negative_counts == ((0, 1),)
    assert metrics.morphology_average_precisions == ((0.5, 0.5),)

    assert not model.training

    for parameter, previous_parameter in zip(
        model.parameters(),
        parameters_before_evaluation,
        strict=True,
    ):
        torch.testing.assert_close(
            parameter,
            previous_parameter,
        )


def test_evaluation_epoch_reports_named_token_slice() -> None:
    model = TinyEpochModel()

    metrics = evaluate_supervised_token_task_epoch(
        model=model,
        batches=(
            _training_batch(0),
            _training_batch(1),
        ),
        device=torch.device("cpu"),
        morphology_schema=MORPHOLOGY_SCHEMA,
        token_slice_masks={
            "rare": (
                torch.tensor([[True]]),
                torch.tensor([[False]]),
            ),
        },
    )

    assert len(metrics.token_slices) == 1
    token_slice = metrics.token_slices[0]
    assert token_slice.name == "rare"
    assert token_slice.metrics.token_count == 1
    assert token_slice.metrics.lemma_target_count == 1
    assert token_slice.metrics.upos_accuracy == 1.0
    assert token_slice.metrics.lemma_rule_accuracy == 1.0
    assert token_slice.metrics.morphology_accuracies == (1.0,)


def test_evaluation_epoch_forwards_decoded_predictions_to_observer() -> None:
    class RecordingObserver:
        def __init__(self) -> None:
            self.predictions: list[TokenTaskPredictionBatch] = []

        def add(self, *, predictions: TokenTaskPredictionBatch) -> None:
            self.predictions.append(predictions)

    observer = RecordingObserver()

    evaluate_supervised_token_task_epoch(
        model=TinyEpochModel(),
        batches=(_training_batch(0), _training_batch(1)),
        device=torch.device("cpu"),
        morphology_schema=MORPHOLOGY_SCHEMA,
        prediction_observers=(observer,),
    )

    assert len(observer.predictions) == 2
    assert observer.predictions[0].upos_ids.tolist() == [[0]]
    assert observer.predictions[1].token_mask.tolist() == [[True]]


def test_evaluation_epoch_applies_optional_morphology_logit_correction() -> None:
    model = TinyEpochModel()
    with torch.no_grad():
        model.morphology.copy_(torch.tensor([0.0, 1.0]))

    metrics = evaluate_supervised_token_task_epoch(
        model=model,
        batches=(_training_batch(0),),
        device=torch.device("cpu"),
        morphology_schema=MORPHOLOGY_SCHEMA,
        morphology_logit_correction=MorphologyLogitCorrection(
            strength=1.0,
            weights=(torch.tensor([1.0, 4.0]),),
        ),
    )

    assert metrics.morphology_accuracies == (1.0,)
    assert math.isclose(metrics.losses.morphology_loss, 1.3132616, rel_tol=1e-6)


def test_training_epoch_forwards_loss_weights() -> None:
    model = TinyEpochModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_step_count=1,
        warmup_ratio=0.0,
    )
    loss_weights = TokenTaskLossWeights(
        morphology_weights=(torch.tensor([1.0, 3.0]),),
    )

    metrics = train_supervised_token_task_epoch(
        model=model,
        batches=(_training_batch(1),),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
        morphology_schema=MORPHOLOGY_SCHEMA,
        loss_weights=loss_weights,
    )

    assert math.isclose(
        metrics.morphology_loss,
        3.0 * math.log(2.0),
        rel_tol=1e-6,
    )


def test_distilled_epoch_reports_both_learning_signals() -> None:
    student = TinyEpochModel()
    teacher = TinyEpochModel()

    with torch.no_grad():
        teacher.upos.copy_(torch.tensor([2.0, -2.0]))
        teacher.morphology.copy_(torch.tensor([1.0, -1.0]))
        teacher.lemma_rules.copy_(torch.tensor([-2.0, 2.0]))

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=0.1,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_step_count=1,
        warmup_ratio=0.0,
    )

    metrics = train_distilled_token_task_epoch(
        student=student,
        teacher=teacher,
        batches=(_training_batch(1),),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=2.0,
            weight=0.5,
        ),
        morphology_schema=MORPHOLOGY_SCHEMA,
    )

    assert metrics.supervised_metrics.batch_count == 1
    assert metrics.distillation_metrics.batch_count == 1
    assert math.isfinite(metrics.supervised_metrics.total_loss)
    assert math.isfinite(metrics.distillation_metrics.total_loss)
    assert math.isfinite(metrics.combined_loss)
    assert optimizer.param_groups[0]["lr"] == 0.0
