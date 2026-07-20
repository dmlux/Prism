import torch
from torch import Tensor


def calculate_average_precision(
    *,
    scores: Tensor,
    targets: Tensor,
) -> float | None:
    if scores.ndim != 1 or targets.ndim != 1:
        raise ValueError("Scores and targets must have one dimension.")

    if scores.shape != targets.shape:
        raise ValueError("Scores and targets must have the same shape.")

    if not scores.is_floating_point():
        raise ValueError("Scores must be floating point.")

    if targets.dtype != torch.bool:
        raise ValueError("Targets must be boolean.")

    if scores.device != targets.device:
        raise ValueError("Scores and targets must use the same device.")

    if not torch.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    positive_count = targets.sum()

    if positive_count.item() == 0:
        return None

    sorted_indices = torch.argsort(
        scores,
        descending=True,
        stable=True,
    )
    sorted_scores = scores[sorted_indices]
    sorted_targets = targets[sorted_indices]

    cumulative_true_positives = sorted_targets.cumsum(
        dim=0,
    )
    cumulative_false_positives = (~sorted_targets).cumsum(dim=0)

    threshold_end_mask = torch.ones_like(
        sorted_targets,
    )
    threshold_end_mask[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    true_positives = cumulative_true_positives[threshold_end_mask].to(torch.float32)
    false_positives = cumulative_false_positives[threshold_end_mask].to(torch.float32)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count.to(torch.float32)

    previous_recall = torch.cat(
        (
            torch.zeros(
                1,
                device=recall.device,
                dtype=recall.dtype,
            ),
            recall[:-1],
        )
    )

    average_precision = ((recall - previous_recall) * precision).sum()

    return float(average_precision.item())
