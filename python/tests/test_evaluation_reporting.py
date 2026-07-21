import pytest

from prism.evaluation.classification import ClassificationMetrics
from prism.evaluation.reporting import (
    format_classification_metric_rows,
    format_morphology_accuracy_rows,
    format_scalar_metric_rows,
)


def test_scalar_metric_rows_align_names_and_values() -> None:
    rows = format_scalar_metric_rows(
        metric_names=(
            "Development loss",
            "UPOS accuracy",
            "Lemma-rule accuracy",
        ),
        values=(0.1735024005, 1.0, 0.9668693542),
    )

    assert rows == (
        "Development loss       0.173502",
        "UPOS accuracy          1.000000",
        "Lemma-rule accuracy    0.966869",
    )


def test_classification_metric_rows_align_labels_support_and_values() -> None:
    rows = format_classification_metric_rows(
        labels=("<NONE>", "Com"),
        metrics=(
            ClassificationMetrics(
                support=21405,
                precision=0.9867,
                recall=0.9773,
                f1=0.982,
            ),
            ClassificationMetrics(
                support=733,
                precision=0.6094,
                recall=0.0532,
                f1=0.0979,
            ),
        ),
        average_precisions=(0.9984, None),
    )

    assert rows == (
        "  <NONE>    support=21405    precision=0.9867    "
        "recall=0.9773    f1=0.9820    average_precision=   0.9984",
        "  Com       support=  733    precision=0.6094    "
        "recall=0.0532    f1=0.0979    average_precision=undefined",
    )


def test_classification_metric_rows_require_matching_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="Classification metrics must match the label count",
    ):
        format_classification_metric_rows(
            labels=("<NONE>",),
            metrics=(),
            average_precisions=(1.0,),
        )


def test_morphology_accuracy_rows_align_names_and_values() -> None:
    rows = format_morphology_accuracy_rows(
        feature_names=("Case", "Polarity"),
        overall_accuracies=(0.987654321, 1.0),
        annotated_accuracies=(1.0, 0.5),
        prefix="Development",
    )

    assert rows == (
        "Development Case        overall=0.987654    annotated=1.000000",
        "Development Polarity    overall=1.000000    annotated=0.500000",
    )


def test_morphology_accuracy_rows_require_matching_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="Overall accuracies must match the feature count",
    ):
        format_morphology_accuracy_rows(
            feature_names=("Case",),
            overall_accuracies=(),
            annotated_accuracies=(1.0,),
            prefix="Development",
        )
