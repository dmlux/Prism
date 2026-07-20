from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassificationMetrics:
    support: int
    precision: float
    recall: float
    f1: float


def calculate_classification_metrics(
    *,
    true_positive_count: int,
    false_positive_count: int,
    false_negative_count: int,
) -> ClassificationMetrics:
    for count in (
        true_positive_count,
        false_positive_count,
        false_negative_count,
    ):
        if count < 0:
            raise ValueError("Classification counts must not be negative.")

    support = true_positive_count + false_negative_count

    precision_denominator = true_positive_count + false_positive_count
    recall_denominator = support
    f1_denominator = (
        2 * true_positive_count + false_positive_count + false_negative_count
    )

    precision = (
        0.0
        if precision_denominator == 0
        else (true_positive_count / precision_denominator)
    )
    recall = (
        0.0 if recall_denominator == 0 else true_positive_count / recall_denominator
    )
    f1 = 0.0 if f1_denominator == 0 else (2 * true_positive_count / f1_denominator)

    return ClassificationMetrics(
        support=support,
        precision=precision,
        recall=recall,
        f1=f1,
    )
