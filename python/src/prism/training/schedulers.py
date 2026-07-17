import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_linear_warmup_decay_scheduler(
    *,
    optimizer: Optimizer,
    total_step_count: int,
    warmup_ratio: float,
) -> LambdaLR:
    if total_step_count <= 0:
        raise ValueError("Total scheduler step count must be positive.")
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("Warmup ratio must be finite and in [0,1).")

    if warmup_ratio == 0.0:
        warmup_step_count = 0
    else:
        warmup_step_count = max(1, int(total_step_count * warmup_ratio))

    decay_step_count = total_step_count - warmup_step_count

    def learning_rate_multiplier(
        completed_step_count: int,
    ) -> float:
        next_step = completed_step_count + 1

        if next_step <= warmup_step_count:
            return next_step / warmup_step_count

        if decay_step_count == 0:
            return 0.0

        remaining_step_count = total_step_count - next_step + 1

        return max(
            0.0,
            remaining_step_count / decay_step_count,
        )

    return LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )
