from dataclasses import dataclass
import math

from prism.modeling.morphology_bundle_reranker import (
    MorphologyBundleLossGradientScope,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedTrainingConfig:
    epoch_count: int
    batch_size: int
    backbone_learning_rate: float
    task_head_learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    warmup_ratio: float
    random_seed: int
    morphology_weight_cap: float | None = None
    morphology_bundle_loss_weight: float = 0.0
    morphology_bundle_loss_gradient_scope: MorphologyBundleLossGradientScope = (
        MorphologyBundleLossGradientScope.FULL
    )
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.epoch_count <= 0:
            raise ValueError("Epoch count must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        for name, learning_rate in (
            (
                "Backbone learning rate",
                self.backbone_learning_rate,
            ),
            (
                "Task-head learning rate",
                self.task_head_learning_rate,
            ),
        ):
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("Weight decay must be finite and non-negative.")
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0:
            raise ValueError("Maximum gradient norm must be finite and positive.")
        if not math.isfinite(self.warmup_ratio) or not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("Warmup ratio must be finite and in [0,1).")
        if self.random_seed < 0:
            raise ValueError("Random seed must be non-negative.")

        if self.morphology_weight_cap is not None and (
            not math.isfinite(self.morphology_weight_cap)
            or self.morphology_weight_cap < 1.0
        ):
            raise ValueError("Morphology weight cap must be finite and at least one.")
        if (
            not math.isfinite(self.morphology_bundle_loss_weight)
            or self.morphology_bundle_loss_weight < 0.0
        ):
            raise ValueError(
                "Morphology bundle loss weight must be finite and non-negative."
            )
        if not isinstance(
            self.morphology_bundle_loss_gradient_scope,
            MorphologyBundleLossGradientScope,
        ):
            raise ValueError("Morphology bundle-loss gradient scope is invalid.")
        if (
            self.morphology_bundle_loss_gradient_scope
            is not MorphologyBundleLossGradientScope.FULL
            and self.morphology_bundle_loss_weight == 0.0
        ):
            raise ValueError(
                "A restricted bundle-loss gradient requires a positive bundle loss "
                "weight."
            )
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("Early-stopping patience must be positive.")
