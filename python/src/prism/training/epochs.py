from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from prism.evaluation.metrics import (
    TokenTaskEvaluationAccumulator,
    TokenTaskEvaluationMetrics,
)
from prism.evaluation.prediction_observer import TokenTaskPredictionObserver
from prism.evaluation.universal_dependencies import (
    UniversalDependenciesEvaluationAccumulator,
    UniversalDependenciesEvaluationMetrics,
)
from prism.modeling.decoding import (
    MorphologyLogitCorrection,
    apply_morphology_logit_correction,
    decode_token_task_logits,
)
from prism.schema import MorphologySchema
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.losses import (
    MorphologyBundleLossPolicy,
    TokenTaskLosses,
    TokenTaskLossWeights,
)
from prism.training.distillation import TokenTaskDistillationPolicy
from prism.training.relation_distillation import RelationDistillationPolicy
from prism.training.steps import (
    evaluate_supervised_token_task_step,
    train_distilled_token_task_step,
    train_supervised_token_task_step,
)

if TYPE_CHECKING:
    from prism.training.silver_batches import SilverTokenTaskBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedEpochMetrics:
    batch_count: int
    token_count: int
    lemma_target_count: int
    upos_loss: float
    morphology_loss: float
    lemma_rule_loss: float
    morphology_bundle_loss: float = 0.0
    morphology_bundle_loss_weight: float = 0.0
    morphology_bundle_target_count: int = 0
    morphology_bundle_token_count: int = 0

    @property
    def total_loss(self) -> float:
        return (
            self.upos_loss
            + self.morphology_loss
            + self.lemma_rule_loss
            + self.morphology_bundle_loss_weight * self.morphology_bundle_loss
        )

    @property
    def morphology_bundle_coverage(self) -> float | None:
        if self.morphology_bundle_token_count == 0:
            return None
        return self.morphology_bundle_target_count / self.morphology_bundle_token_count


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedEvaluationMetrics:
    losses: SupervisedEpochMetrics
    upos_accuracy: float
    morphology_bundle_exact_accuracy: float | None
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
    universal_dependencies: UniversalDependenciesEvaluationMetrics | None = None
    token_slices: tuple["NamedTokenTaskEvaluationMetrics", ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedTokenTaskEvaluationMetrics:
    name: str
    metrics: TokenTaskEvaluationMetrics
    universal_dependencies: UniversalDependenciesEvaluationMetrics | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("Evaluation slice name must be non-empty and trimmed.")


@dataclass(frozen=True, slots=True, kw_only=True)
class DistilledEpochMetrics:
    supervised_metrics: SupervisedEpochMetrics
    distillation_metrics: SupervisedEpochMetrics
    combined_loss: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MixedEpochMetrics:
    """Gold plus silver training signals of one interleaved epoch."""

    gold_metrics: "SupervisedEpochMetrics | DistilledEpochMetrics"
    silver_batch_count: int
    silver_token_count: int
    silver_loss_weight: float
    silver_upos_loss: float
    silver_morphology_loss: float
    silver_lemma_rule_loss: float
    relation_loss: float | None = None

    @property
    def silver_total_loss(self) -> float:
        return (
            self.silver_upos_loss
            + self.silver_morphology_loss
            + self.silver_lemma_rule_loss
        )


@dataclass(slots=True, kw_only=True)
class _EpochLossAccumulator:
    device: torch.device
    batch_count: int = 0
    token_count: torch.Tensor = field(init=False)
    lemma_target_count: torch.Tensor = field(init=False)
    upos_loss_sum: torch.Tensor = field(init=False)
    morphology_loss_sum: torch.Tensor = field(init=False)
    lemma_rule_loss_sum: torch.Tensor = field(init=False)
    morphology_bundle_loss_sum: torch.Tensor = field(init=False)
    morphology_bundle_target_count: torch.Tensor = field(init=False)
    morphology_bundle_token_count: torch.Tensor = field(init=False)
    morphology_bundle_loss_weight: float = 0.0
    morphology_bundle_loss_active: bool = False

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
        self.morphology_bundle_loss_sum = torch.zeros((), device=self.device)
        self.morphology_bundle_target_count = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        self.morphology_bundle_token_count = torch.zeros(
            (),
            dtype=torch.long,
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
        if losses.morphology_bundle_loss is not None:
            if (
                self.morphology_bundle_loss_active
                and losses.morphology_bundle_loss_weight
                != self.morphology_bundle_loss_weight
            ):
                raise ValueError(
                    "Morphology bundle loss weight must be constant within an epoch."
                )
            self.morphology_bundle_loss_weight = losses.morphology_bundle_loss_weight
            self.morphology_bundle_loss_active = True
            self.morphology_bundle_loss_sum += (
                losses.morphology_bundle_loss * losses.morphology_bundle_target_count
            )
            self.morphology_bundle_target_count += losses.morphology_bundle_target_count
            self.morphology_bundle_token_count += losses.morphology_bundle_token_count

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
        morphology_bundle_target_count = int(self.morphology_bundle_target_count.item())
        morphology_bundle_token_count = int(self.morphology_bundle_token_count.item())
        morphology_bundle_loss = (
            0.0
            if morphology_bundle_target_count == 0
            else (
                self.morphology_bundle_loss_sum / self.morphology_bundle_target_count
            ).item()
        )

        return SupervisedEpochMetrics(
            batch_count=self.batch_count,
            token_count=token_count,
            lemma_target_count=lemma_target_count,
            upos_loss=(self.upos_loss_sum / self.token_count).item(),
            morphology_loss=(self.morphology_loss_sum / self.token_count).item(),
            lemma_rule_loss=lemma_rule_loss,
            morphology_bundle_loss=morphology_bundle_loss,
            morphology_bundle_loss_weight=self.morphology_bundle_loss_weight,
            morphology_bundle_target_count=morphology_bundle_target_count,
            morphology_bundle_token_count=morphology_bundle_token_count,
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
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
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
            morphology_bundle_loss_policy=morphology_bundle_loss_policy,
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
    distillation_policy: TokenTaskDistillationPolicy,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
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
            distillation_policy=distillation_policy,
            morphology_schema=morphology_schema,
            loss_weights=loss_weights,
            morphology_bundle_loss_policy=morphology_bundle_loss_policy,
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
            + distillation_policy.upos_weight * distillation_metrics.upos_loss
            + distillation_policy.morphology_weight
            * distillation_metrics.morphology_loss
            + distillation_policy.lemma_rule_weight
            * distillation_metrics.lemma_rule_loss
        ),
    )


def train_mixed_token_task_epoch(
    *,
    student: nn.Module,
    teacher: nn.Module | None,
    gold_batches: Iterable[SupervisedTokenTaskBatch],
    silver_batches: Iterable["SilverTokenTaskBatch"],
    gold_batch_count: int,
    silver_batch_count: int,
    order_seed: int,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    max_gradient_norm: float,
    morphology_schema: MorphologySchema,
    silver_loss_weight: float,
    distillation_policy: TokenTaskDistillationPolicy | None = None,
    loss_weights: TokenTaskLossWeights | None = None,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
    relation_teacher: nn.Module | None = None,
    relation_policy: "RelationDistillationPolicy | None" = None,
) -> MixedEpochMetrics:
    """Interleave gold and silver batches deterministically in one epoch.

    Gold batches keep their existing supervised (and optional teacher
    distillation) objective; silver batches contribute the soft-target KD
    loss. The interleaving order is a seeded permutation, so gold and silver
    signals alternate instead of one regime dominating the end of the epoch.
    """

    from prism.training.silver_training import train_silver_kd_step

    if gold_batch_count <= 0 or silver_batch_count <= 0:
        raise ValueError("Mixed epochs require gold and silver batches.")
    if (teacher is None) != (distillation_policy is None):
        raise ValueError(
            "Gold distillation requires both the teacher and its policy."
        )
    if (relation_teacher is None) != (relation_policy is None):
        raise ValueError(
            "Relation distillation requires both the teacher and its policy."
        )
    if relation_teacher is not None and teacher is None:
        raise ValueError(
            "Relation distillation extends the gold distillation step and "
            "therefore requires the gold distillation teacher."
        )

    student.to(device)
    if teacher is not None:
        teacher.to(device)
    if relation_teacher is not None:
        relation_teacher.to(device)

    generator = torch.Generator()
    generator.manual_seed(order_seed)
    order = torch.cat(
        (
            torch.zeros(gold_batch_count, dtype=torch.long),
            torch.ones(silver_batch_count, dtype=torch.long),
        )
    )
    order = order[torch.randperm(len(order), generator=generator)]

    gold_iterator = iter(gold_batches)
    silver_iterator = iter(silver_batches)
    supervised_accumulator = _EpochLossAccumulator(device=device)
    distillation_accumulator = (
        None if teacher is None else _EpochLossAccumulator(device=device)
    )
    silver_upos_loss_sum = torch.zeros((), device=device)
    silver_morphology_loss_sum = torch.zeros((), device=device)
    silver_lemma_loss_sum = torch.zeros((), device=device)
    silver_upos_tokens = 0
    silver_morphology_tokens = 0
    silver_lemma_tokens = 0
    silver_token_count = 0
    relation_loss_sum = torch.zeros((), device=device)
    relation_batch_count = 0

    for kind in order.tolist():
        if kind == 0:
            gold_batch = next(gold_iterator).to(device)
            if teacher is None:
                losses = train_supervised_token_task_step(
                    model=student,
                    batch=gold_batch,
                    optimizer=optimizer,
                    max_gradient_norm=max_gradient_norm,
                    morphology_schema=morphology_schema,
                    loss_weights=loss_weights,
                    morphology_bundle_loss_policy=morphology_bundle_loss_policy,
                )
                supervised_accumulator.add(batch=gold_batch, losses=losses)
            else:
                assert distillation_policy is not None
                assert distillation_accumulator is not None
                distilled_losses = train_distilled_token_task_step(
                    student=student,
                    teacher=teacher,
                    batch=gold_batch,
                    optimizer=optimizer,
                    max_gradient_norm=max_gradient_norm,
                    distillation_policy=distillation_policy,
                    morphology_schema=morphology_schema,
                    loss_weights=loss_weights,
                    morphology_bundle_loss_policy=morphology_bundle_loss_policy,
                    relation_teacher=relation_teacher,
                    relation_policy=relation_policy,
                )
                supervised_accumulator.add(
                    batch=gold_batch,
                    losses=distilled_losses.supervised_losses,
                )
                distillation_accumulator.add(
                    batch=gold_batch,
                    losses=distilled_losses.distillation_losses,
                )
                if distilled_losses.relation_loss is not None:
                    relation_loss_sum += distilled_losses.relation_loss
                    relation_batch_count += 1
        else:
            silver_batch = next(silver_iterator).to(device)
            silver_losses = train_silver_kd_step(
                model=student,
                batch=silver_batch,
                optimizer=optimizer,
                max_gradient_norm=max_gradient_norm,
                morphology_schema=morphology_schema,
                silver_loss_weight=silver_loss_weight,
            )
            silver_upos_loss_sum += (
                silver_losses.upos_loss.detach()
                * silver_losses.upos_token_count
            )
            silver_morphology_loss_sum += (
                silver_losses.morphology_loss.detach()
                * silver_losses.morphology_token_count
            )
            silver_lemma_loss_sum += (
                silver_losses.lemma_rule_loss.detach()
                * silver_losses.lemma_token_count
            )
            silver_upos_tokens += silver_losses.upos_token_count
            silver_morphology_tokens += silver_losses.morphology_token_count
            silver_lemma_tokens += silver_losses.lemma_token_count
            silver_token_count += int(
                silver_batch.model_inputs.token_mask.sum().item()
            )
        scheduler.step()

    supervised_metrics = supervised_accumulator.finish(
        empty_epoch_message="Mixed epoch must contain gold batches.",
    )
    gold_metrics: SupervisedEpochMetrics | DistilledEpochMetrics
    if distillation_accumulator is None:
        gold_metrics = supervised_metrics
    else:
        assert distillation_policy is not None
        distillation_metrics = distillation_accumulator.finish(
            empty_epoch_message="Mixed epoch must contain gold batches.",
        )
        gold_metrics = DistilledEpochMetrics(
            supervised_metrics=supervised_metrics,
            distillation_metrics=distillation_metrics,
            combined_loss=(
                supervised_metrics.total_loss
                + distillation_policy.upos_weight * distillation_metrics.upos_loss
                + distillation_policy.morphology_weight
                * distillation_metrics.morphology_loss
                + distillation_policy.lemma_rule_weight
                * distillation_metrics.lemma_rule_loss
            ),
        )

    return MixedEpochMetrics(
        gold_metrics=gold_metrics,
        silver_batch_count=silver_batch_count,
        silver_token_count=silver_token_count,
        silver_loss_weight=silver_loss_weight,
        silver_upos_loss=(
            0.0
            if silver_upos_tokens == 0
            else float(silver_upos_loss_sum.item()) / silver_upos_tokens
        ),
        silver_morphology_loss=(
            0.0
            if silver_morphology_tokens == 0
            else float(silver_morphology_loss_sum.item()) / silver_morphology_tokens
        ),
        silver_lemma_rule_loss=(
            0.0
            if silver_lemma_tokens == 0
            else float(silver_lemma_loss_sum.item()) / silver_lemma_tokens
        ),
        relation_loss=(
            None
            if relation_batch_count == 0
            else float(relation_loss_sum.item()) / relation_batch_count
        ),
    )


def evaluate_supervised_token_task_epoch(
    *,
    model: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    device: torch.device,
    morphology_schema: MorphologySchema,
    token_slice_masks: Mapping[str, Sequence[torch.Tensor]] | None = None,
    universal_dependencies_accumulator: (
        UniversalDependenciesEvaluationAccumulator | None
    ) = None,
    prediction_observers: Sequence[TokenTaskPredictionObserver] = (),
    morphology_logit_correction: MorphologyLogitCorrection | None = None,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
) -> SupervisedEvaluationMetrics:
    resolved_slice_masks = {} if token_slice_masks is None else token_slice_masks
    if any(not name or name.strip() != name for name in resolved_slice_masks):
        raise ValueError("Evaluation slice names must be non-empty and trimmed.")

    overall_accumulator = TokenTaskEvaluationAccumulator(
        device=device,
        morphology_schema=morphology_schema,
    )
    slice_accumulators = {
        name: TokenTaskEvaluationAccumulator(
            device=device,
            morphology_schema=morphology_schema,
        )
        for name in resolved_slice_masks
    }
    slice_universal_dependencies_accumulators = {
        name: (
            None
            if universal_dependencies_accumulator is None
            else universal_dependencies_accumulator.spawn_empty()
        )
        for name in resolved_slice_masks
    }
    batch_index = 0

    def process_batch(
        batch: SupervisedTokenTaskBatch,
    ) -> TokenTaskLosses:
        nonlocal batch_index

        logits, losses = evaluate_supervised_token_task_step(
            model=model,
            batch=batch,
            morphology_schema=morphology_schema,
            morphology_bundle_loss_policy=morphology_bundle_loss_policy,
        )
        prediction_logits = (
            logits
            if morphology_logit_correction is None
            else apply_morphology_logit_correction(
                logits=logits,
                morphology_schema=morphology_schema,
                correction=morphology_logit_correction,
            )
        )
        predictions = decode_token_task_logits(
            logits=prediction_logits,
            token_mask=batch.targets.token_mask,
            morphology_schema=morphology_schema,
        )
        overall_accumulator.add(
            logits=prediction_logits,
            predictions=predictions,
            targets=batch.targets,
            evaluation_mask=batch.targets.token_mask,
        )
        if universal_dependencies_accumulator is not None:
            universal_dependencies_accumulator.add(predictions=predictions)
        for observer in prediction_observers:
            observer.add(predictions=predictions)

        for name, masks in resolved_slice_masks.items():
            if batch_index >= len(masks):
                raise ValueError(
                    f"Evaluation slice {name!r} contains fewer masks than batches."
                )

            evaluation_mask = masks[batch_index].to(device=device)
            slice_accumulators[name].add(
                logits=prediction_logits,
                predictions=predictions,
                targets=batch.targets,
                evaluation_mask=evaluation_mask,
            )
            slice_universal_dependencies_accumulator = (
                slice_universal_dependencies_accumulators[name]
            )
            if slice_universal_dependencies_accumulator is not None:
                slice_universal_dependencies_accumulator.add(
                    predictions=predictions,
                    evaluation_mask=evaluation_mask,
                )

        batch_index += 1
        return losses

    losses = _run_supervised_token_task_epoch(
        model=model,
        batches=batches,
        device=device,
        process_batch=process_batch,
        empty_epoch_message=("Evaluation epoch must contain batches."),
    )

    for name, masks in resolved_slice_masks.items():
        if len(masks) != batch_index:
            raise ValueError(
                f"Evaluation slice {name!r} contains more masks than batches."
            )

    overall_metrics = overall_accumulator.finish(
        empty_slice_message="Evaluation epoch must contain tokens.",
    )
    if overall_metrics.token_count != losses.token_count:
        raise RuntimeError("Evaluation metric and loss token counts must match.")
    if overall_metrics.lemma_target_count != losses.lemma_target_count:
        raise RuntimeError("Evaluation metric and loss lemma counts must match.")

    return SupervisedEvaluationMetrics(
        losses=losses,
        upos_accuracy=overall_metrics.upos_accuracy,
        morphology_bundle_exact_accuracy=(
            overall_metrics.morphology_bundle_exact_accuracy
        ),
        morphology_accuracies=overall_metrics.morphology_accuracies,
        morphology_annotated_accuracies=(
            overall_metrics.morphology_annotated_accuracies
        ),
        lemma_rule_accuracy=overall_metrics.lemma_rule_accuracy,
        morphology_true_positive_counts=(
            overall_metrics.morphology_true_positive_counts
        ),
        morphology_false_positive_counts=(
            overall_metrics.morphology_false_positive_counts
        ),
        morphology_false_negative_counts=(
            overall_metrics.morphology_false_negative_counts
        ),
        morphology_average_precisions=(overall_metrics.morphology_average_precisions),
        universal_dependencies=(
            None
            if universal_dependencies_accumulator is None
            else universal_dependencies_accumulator.finish()
        ),
        token_slices=tuple(
            NamedTokenTaskEvaluationMetrics(
                name=name,
                metrics=accumulator.finish(
                    empty_slice_message=(
                        f"Evaluation slice {name!r} must select at least one token."
                    ),
                ),
                universal_dependencies=(
                    None
                    if slice_universal_dependencies_accumulators[name] is None
                    else slice_universal_dependencies_accumulators[name].finish()
                ),
            )
            for name, accumulator in slice_accumulators.items()
        ),
    )
