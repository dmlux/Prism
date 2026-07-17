from dataclasses import dataclass
import math


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
