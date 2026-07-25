import pytest

from prism.evaluation.classification import ClassificationMetrics
from prism.evaluation.metrics import TokenTaskEvaluationMetrics
from prism.evaluation.universal_dependencies import (
    UniversalDependenciesEvaluationMetrics,
    UniversalDependenciesMetricScore,
)
from prism.evaluation.reporting import (
    format_classification_metric_rows,
    format_morphology_accuracy_rows,
    format_morphology_error_attribution_rows,
    format_scalar_metric_rows,
    format_token_slice_metric_rows,
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


def test_token_slice_metric_rows_align_summary() -> None:
    rows = format_token_slice_metric_rows(
        slice_name="OOV",
        metrics=TokenTaskEvaluationMetrics(
            token_count=12,
            lemma_target_count=10,
            lemma_annotation_count=12,
            upos_accuracy=0.75,
            morphology_bundle_exact_accuracy=0.75,
            morphology_correct_counts=(6,),
            morphology_accuracies=(0.5,),
            morphology_annotated_accuracies=(0.4,),
            lemma_rule_accuracy=0.6,
            lemma_rule_coverage=10 / 12,
            lemma_end_to_end_accuracy=0.5,
            morphology_true_positive_counts=((0, 3),),
            morphology_false_positive_counts=((0, 1),),
            morphology_false_negative_counts=((0, 2),),
            morphology_average_precisions=((None, 0.7),),
        ),
    )

    assert rows == (
        "OOV tokens                        12",
        "OOV UPOS accuracy                 0.750000",
        "OOV lemma annotations             12",
        "OOV lemma-rule targets            10",
        "OOV lemma-rule coverage           0.833333",
        "OOV lemma-rule accuracy           0.600000",
        "OOV lemma end-to-end accuracy     0.500000",
        "OOV morphology micro precision    0.750000",
        "OOV morphology micro recall       0.600000",
        "OOV morphology micro F1           0.666667",
    )


def test_token_slice_metric_rows_include_ud_scores() -> None:
    score = UniversalDependenciesMetricScore(
        gold_total=12,
        system_total=12,
        correct=9,
        aligned_total=12,
    )

    rows = format_token_slice_metric_rows(
        slice_name="RARE",
        metrics=TokenTaskEvaluationMetrics(
            token_count=12,
            lemma_target_count=10,
            lemma_annotation_count=12,
            upos_accuracy=0.75,
            morphology_bundle_exact_accuracy=0.75,
            morphology_correct_counts=(6,),
            morphology_accuracies=(0.5,),
            morphology_annotated_accuracies=(0.4,),
            lemma_rule_accuracy=0.6,
            lemma_rule_coverage=10 / 12,
            lemma_end_to_end_accuracy=0.5,
            morphology_true_positive_counts=((0, 3),),
            morphology_false_positive_counts=((0, 1),),
            morphology_false_negative_counts=((0, 2),),
            morphology_average_precisions=((None, 0.7),),
        ),
        universal_dependencies=UniversalDependenciesEvaluationMetrics(
            upos=score,
            ufeats=score,
            lemmas=score,
        ),
    )

    assert rows[2:5] == (
        "RARE UD UPOS F1                    0.750000",
        "RARE UD UFeats F1                  0.750000",
        "RARE UD Lemmas F1                  0.750000",
    )


def test_morphology_error_attribution_rows_rank_exact_feature_errors() -> None:
    rows = format_morphology_error_attribution_rows(
        slice_name="OOV",
        feature_names=("Case", "Gender", "Polarity"),
        metrics=TokenTaskEvaluationMetrics(
            token_count=10,
            lemma_target_count=0,
            lemma_annotation_count=0,
            upos_accuracy=1.0,
            morphology_bundle_exact_accuracy=1.0,
            morphology_correct_counts=(8, 5, 10),
            morphology_accuracies=(0.8, 0.5, 1.0),
            morphology_annotated_accuracies=(0.5, 0.5, None),
            lemma_rule_accuracy=None,
            lemma_rule_coverage=None,
            lemma_end_to_end_accuracy=None,
            morphology_true_positive_counts=((0,), (0,), (0,)),
            morphology_false_positive_counts=((0,), (0,), (0,)),
            morphology_false_negative_counts=((0,), (0,), (0,)),
            morphology_average_precisions=((None,), (None,), (None,)),
        ),
    )

    assert rows == (
        "OOV Gender    errors=5    share= 71.43%    accuracy=0.500000",
        "OOV Case      errors=2    share= 28.57%    accuracy=0.800000",
    )
