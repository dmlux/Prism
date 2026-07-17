import pytest
import torch

from prism.training import (
    build_linear_warmup_decay_scheduler,
)


def test_scheduler_warms_up_and_then_decays_linearly() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=1.0,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_step_count=6,
        warmup_ratio=1 / 3,
    )

    learning_rates: list[float] = []

    for _ in range(6):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert learning_rates == pytest.approx(
        [
            0.5,
            1.0,
            1.0,
            0.75,
            0.5,
            0.25,
        ]
    )
    assert optimizer.param_groups[0]["lr"] == 0.0
