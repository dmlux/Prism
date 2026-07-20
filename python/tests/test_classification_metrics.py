from prism.evaluation.classification import (
    calculate_classification_metrics,
)


def test_calculate_classification_metrics() -> None:
    metrics = calculate_classification_metrics(
        true_positive_count=2,
        false_positive_count=1,
        false_negative_count=3,
    )

    assert metrics.support == 5
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 5
    assert metrics.f1 == 0.5
