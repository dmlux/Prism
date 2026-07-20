import math

import torch

from prism.evaluation.ranking import (
    calculate_average_precision,
)


def test_calculate_average_precision_uses_positive_ranks() -> None:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
    targets = torch.tensor([True, False, True, False])

    average_precision = calculate_average_precision(
        scores=scores,
        targets=targets,
    )

    assert math.isclose(
        average_precision,
        5.0 / 6.0,
        rel_tol=1e-6,
    )
