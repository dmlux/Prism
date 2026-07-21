from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from prism.evaluation.metrics import count_token_task_predictions
from prism.evaluation.ranking import (
    calculate_average_precision,
)
from prism.modeling.decoding import (
    decode_token_task_logits,
    morphology_label_scores,
)
from prism.schema import MorphologySchema
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.losses import (
    TokenTaskLosses,
    TokenTaskLossWeights,
)
from prism.training.steps import (
    evaluate_supervised_token_task_step,
    train_distilled_token_task_step,
    train_supervised_token_task_step,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedEpochMetrics:
    batch_count: int
    token_count: int
    lemma_target_count: int
    upos_loss: float
    morphology_loss: float
    lemma_rule_loss: float

    @property
    def total_loss(self) -> float:
        return self.upos_loss + self.morphology_loss + self.lemma_rule_loss


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedEvaluationMetrics:
    losses: SupervisedEpochMetrics
    upos_accuracy: float
    morphology_accuracies: tuple[float, ...]
    morphology_annotated_accuracies: tuple[
        float | None,
        ...,
    ]
    lemma_rule_accuracy: float | None
    morphology_true_positive_counts: tuple[
        tuple[int, ...],
        ...,
    ]
    morphology_false_positive_counts: tuple[
        tuple[int, ...],
        ...,
    ]
    morphology_false_negative_counts: tuple[
        tuple[int, ...],
        ...,
    ]
    morphology_average_precisions: tuple[
        tuple[float | None, ...],
        ...,
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class DistilledEpochMetrics:
    supervised_metrics: SupervisedEpochMetrics
    distillation_metrics: SupervisedEpochMetrics
    combined_loss: float


@dataclass(slots=True, kw_only=True)
class _EpochLossAccumulator:
    device: torch.device
    batch_count: int = 0
    token_count: torch.Tensor = field(init=False)
    lemma_target_count: torch.Tensor = field(init=False)
    upos_loss_sum: torch.Tensor = field(init=False)
    morphology_loss_sum: torch.Tensor = field(init=False)
    lemma_rule_loss_sum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.lemma_target_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.upos_loss_sum = torch.zeros((), device=self.device)
        self.morphology_loss_sum = torch.zeros(
            (),
            device=self.device,
        )
        self.lemma_rule_loss_sum = torch.zeros(
            (),
            device=self.device,
        )

    def add(
        self,
        *,
        batch: SupervisedTokenTaskBatch,
        losses: TokenTaskLosses,
    ) -> None:
        current_token_count = batch.targets.token_mask.sum()
        current_lemma_target_count = (
            batch.targets.token_mask & batch.targets.lemma_rule_mask
        ).sum()

        self.batch_count += 1
        self.token_count += current_token_count
        self.lemma_target_count += current_lemma_target_count
        self.upos_loss_sum += losses.upos_loss * current_token_count
        self.morphology_loss_sum += losses.morphology_loss * current_token_count
        self.lemma_rule_loss_sum += losses.lemma_rule_loss * current_lemma_target_count

    def finish(
        self,
        *,
        empty_epoch_message: str,
    ) -> SupervisedEpochMetrics:
        if self.batch_count == 0:
            raise ValueError(empty_epoch_message)

        token_count = int(self.token_count.item())
        lemma_target_count = int(self.lemma_target_count.item())

        lemma_rule_loss = (
            0.0
            if lemma_target_count == 0
            else (self.lemma_rule_loss_sum / self.lemma_target_count).item()
        )

        return SupervisedEpochMetrics(
            batch_count=self.batch_count,
            token_count=token_count,
            lemma_target_count=lemma_target_count,
            upos_loss=(self.upos_loss_sum / self.token_count).item(),
            morphology_loss=(self.morphology_loss_sum / self.token_count).item(),
            lemma_rule_loss=lemma_rule_loss,
        )


def _run_supervised_token_task_epoch(
    *,
    model: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    device: torch.device,
    process_batch: Callable[
        [SupervisedTokenTaskBatch],
        TokenTaskLosses,
    ],
    empty_epoch_message: str,
) -> SupervisedEpochMetrics:
    model.to(device)
    accumulator = _EpochLossAccumulator(device=device)

    for batch in batches:
        device_batch = batch.to(device)
        losses = process_batch(device_batch)
        accumulator.add(
            batch=device_batch,
            losses=losses,
        )

    return accumulator.finish(
        empty_epoch_message=empty_epoch_message,
    )


def train_supervised_token_task_epoch(
    *,
    model: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    max_gradient_norm: float,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
) -> SupervisedEpochMetrics:
    def process_batch(
        batch: SupervisedTokenTaskBatch,
    ) -> TokenTaskLosses:
        losses = train_supervised_token_task_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
            morphology_schema=morphology_schema,
            loss_weights=loss_weights,
        )
        scheduler.step()
        return losses

    return _run_supervised_token_task_epoch(
        model=model,
        batches=batches,
        device=device,
        process_batch=process_batch,
        empty_epoch_message=("Training epoch must contain batches."),
    )


def train_distilled_token_task_epoch(
    *,
    student: nn.Module,
    teacher: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    max_gradient_norm: float,
    temperature: float,
    distillation_weight: float,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
) -> DistilledEpochMetrics:
    student.to(device)
    teacher.to(device)

    supervised_accumulator = _EpochLossAccumulator(device=device)
    distillation_accumulator = _EpochLossAccumulator(device=device)

    for batch in batches:
        device_batch = batch.to(device)

        losses = train_distilled_token_task_step(
            student=student,
            teacher=teacher,
            batch=device_batch,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
            temperature=temperature,
            distillation_weight=distillation_weight,
            morphology_schema=morphology_schema,
            loss_weights=loss_weights,
        )
        scheduler.step()

        supervised_accumulator.add(
            batch=device_batch,
            losses=losses.supervised_losses,
        )
        distillation_accumulator.add(
            batch=device_batch,
            losses=losses.distillation_losses,
        )

    supervised_metrics = supervised_accumulator.finish(
        empty_epoch_message=("Distillation epoch must contain batches."),
    )
    distillation_metrics = distillation_accumulator.finish(
        empty_epoch_message=("Distillation epoch must contain batches."),
    )

    return DistilledEpochMetrics(
        supervised_metrics=supervised_metrics,
        distillation_metrics=distillation_metrics,
        combined_loss=(
            supervised_metrics.total_loss
            + distillation_weight * distillation_metrics.total_loss
        ),
    )


def evaluate_supervised_token_task_epoch(
    *,
    model: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    device: torch.device,
    morphology_schema: MorphologySchema,
) -> SupervisedEvaluationMetrics:
    upos_correct_count = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )
    morphology_correct_counts = tuple(
        torch.zeros((), dtype=torch.long, device=device)
        for _ in morphology_schema.features
    )
    morphology_annotated_counts = tuple(
        torch.zeros((), dtype=torch.long, device=device)
        for _ in morphology_schema.features
    )
    morphology_annotated_correct_counts = tuple(
        torch.zeros((), dtype=torch.long, device=device)
        for _ in morphology_schema.features
    )
    lemma_rule_correct_count = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )
    morphology_true_positive_counts = tuple(
        torch.zeros(len(feature.labels), dtype=torch.long, device=device)
        for feature in morphology_schema.features
    )
    morphology_false_positive_counts = tuple(
        torch.zeros(len(feature.labels), dtype=torch.long, device=device)
        for feature in morphology_schema.features
    )
    morphology_false_negative_counts = tuple(
        torch.zeros(
            len(feature.labels),
            dtype=torch.long,
            device=device,
        )
        for feature in morphology_schema.features
    )
    morphology_score_batches: tuple[
        list[torch.Tensor],
        ...,
    ] = tuple([] for _ in morphology_schema.features)
    morphology_target_batches: tuple[
        list[torch.Tensor],
        ...,
    ] = tuple([] for _ in morphology_schema.features)

    def process_batch(
        batch: SupervisedTokenTaskBatch,
    ) -> TokenTaskLosses:
        logits, losses = evaluate_supervised_token_task_step(
            model=model,
            batch=batch,
            morphology_schema=morphology_schema,
        )
        predictions = decode_token_task_logits(
            logits=logits,
            token_mask=batch.targets.token_mask,
            morphology_schema=morphology_schema,
        )
        counts = count_token_task_predictions(
            predictions=predictions,
            targets=batch.targets,
        )

        for (
            score_batches,
            target_batches,
            feature_logits,
            feature_targets,
            feature_schema,
        ) in zip(
            morphology_score_batches,
            morphology_target_batches,
            logits.morphology_logits,
            batch.targets.morphology_targets,
            morphology_schema.features,
            strict=True,
        ):
            score_batches.append(
                morphology_label_scores(
                    feature_logits=feature_logits,
                    feature_schema=feature_schema,
                )[batch.targets.token_mask]
                .detach()
                .cpu()
            )
            target_batches.append(
                feature_targets[batch.targets.token_mask].detach().cpu()
            )

        upos_correct_count.add_(counts.upos_correct_count)
        lemma_rule_correct_count.add_(counts.lemma_rule_correct_count)

        for total, current in zip(
            morphology_correct_counts,
            counts.morphology_correct_counts,
            strict=True,
        ):
            total.add_(current)

        for total, current in zip(
            morphology_annotated_counts,
            counts.morphology_annotated_counts,
            strict=True,
        ):
            total.add_(current)

        for total, current in zip(
            morphology_annotated_correct_counts,
            counts.morphology_annotated_correct_counts,
            strict=True,
        ):
            total.add_(current)

        for total, current in zip(
            morphology_true_positive_counts,
            counts.morphology_true_positive_counts,
            strict=True,
        ):
            total.add_(current)

        for total, current in zip(
            morphology_false_positive_counts,
            counts.morphology_false_positive_counts,
            strict=True,
        ):
            total.add_(current)

        for total, current in zip(
            morphology_false_negative_counts,
            counts.morphology_false_negative_counts,
            strict=True,
        ):
            total.add_(current)

        return losses

    losses = _run_supervised_token_task_epoch(
        model=model,
        batches=batches,
        device=device,
        process_batch=process_batch,
        empty_epoch_message=("Evaluation epoch must contain batches."),
    )

    morphology_average_precisions: list[tuple[float | None, ...]] = []

    for score_batches, target_batches in zip(
        morphology_score_batches,
        morphology_target_batches,
        strict=True,
    ):
        feature_scores = torch.cat(
            score_batches,
            dim=0,
        )
        feature_targets = torch.cat(
            target_batches,
            dim=0,
        )

        morphology_average_precisions.append(
            tuple(
                calculate_average_precision(
                    scores=feature_scores[:, label_index],
                    targets=feature_targets[:, label_index],
                )
                for label_index in range(feature_scores.shape[-1])
            )
        )

    morphology_annotated_accuracies = tuple(
        None
        if annotated_count.item() == 0
        else (correct_count / annotated_count).item()
        for correct_count, annotated_count in zip(
            morphology_annotated_correct_counts,
            morphology_annotated_counts,
            strict=True,
        )
    )

    if losses.lemma_target_count == 0:
        lemma_rule_accuracy = None
    else:
        lemma_rule_accuracy = (
            lemma_rule_correct_count / losses.lemma_target_count
        ).item()

    return SupervisedEvaluationMetrics(
        losses=losses,
        upos_accuracy=(upos_correct_count / losses.token_count).item(),
        morphology_accuracies=tuple(
            (correct_count / losses.token_count).item()
            for correct_count in morphology_correct_counts
        ),
        morphology_annotated_accuracies=morphology_annotated_accuracies,
        lemma_rule_accuracy=lemma_rule_accuracy,
        morphology_true_positive_counts=tuple(
            tuple(int(value) for value in counts.detach().cpu().tolist())
            for counts in morphology_true_positive_counts
        ),
        morphology_false_positive_counts=tuple(
            tuple(int(value) for value in counts.detach().cpu().tolist())
            for counts in morphology_false_positive_counts
        ),
        morphology_false_negative_counts=tuple(
            tuple(int(value) for value in counts.detach().cpu().tolist())
            for counts in morphology_false_negative_counts
        ),
        morphology_average_precisions=tuple(morphology_average_precisions),
    )
