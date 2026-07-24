from collections.abc import Sequence

from prism.evaluation.classification import ClassificationMetrics
from prism.evaluation.metrics import TokenTaskEvaluationMetrics
from prism.evaluation.universal_dependencies import (
    UniversalDependenciesEvaluationMetrics,
)


def format_scalar_metric_rows(
    *,
    metric_names: Sequence[str],
    values: Sequence[float],
) -> tuple[str, ...]:
    if not metric_names:
        raise ValueError("Metric report must contain metrics.")
    if len(values) != len(metric_names):
        raise ValueError("Metric values must match the metric count.")
    if any(not name or name.strip() != name for name in metric_names):
        raise ValueError("Metric names must be non-empty and trimmed.")

    metric_name_width = max(len(name) for name in metric_names)

    return tuple(
        f"{name:<{metric_name_width}}    {value:.6f}"
        for name, value in zip(metric_names, values, strict=True)
    )


def format_classification_metric_rows(
    *,
    labels: Sequence[str],
    metrics: Sequence[ClassificationMetrics],
    average_precisions: Sequence[float | None],
    indentation: str = "  ",
) -> tuple[str, ...]:
    if not labels:
        raise ValueError("Classification report must contain labels.")
    if len(metrics) != len(labels):
        raise ValueError("Classification metrics must match the label count.")
    if len(average_precisions) != len(labels):
        raise ValueError("Average precisions must match the label count.")
    if any(not label or label.strip() != label for label in labels):
        raise ValueError("Classification labels must be non-empty and trimmed.")
    if indentation.strip():
        raise ValueError("Classification indentation must contain only whitespace.")

    label_width = max(len(label) for label in labels)
    support_width = max(len(str(label_metrics.support)) for label_metrics in metrics)
    average_precision_texts = tuple(
        "undefined" if value is None else f"{value:.4f}" for value in average_precisions
    )
    average_precision_width = max(len(value) for value in average_precision_texts)

    return tuple(
        f"{indentation}{label:<{label_width}}    "
        f"support={label_metrics.support:>{support_width}}    "
        f"precision={label_metrics.precision:.4f}    "
        f"recall={label_metrics.recall:.4f}    "
        f"f1={label_metrics.f1:.4f}    "
        f"average_precision={average_precision:>{average_precision_width}}"
        for label, label_metrics, average_precision in zip(
            labels,
            metrics,
            average_precision_texts,
            strict=True,
        )
    )


def format_morphology_accuracy_rows(
    *,
    feature_names: Sequence[str],
    overall_accuracies: Sequence[float],
    annotated_accuracies: Sequence[float],
    prefix: str,
) -> tuple[str, ...]:
    if not feature_names:
        raise ValueError("Morphology report must contain features.")
    if not prefix or prefix.strip() != prefix:
        raise ValueError("Morphology report prefix must be non-empty and trimmed.")
    if len(overall_accuracies) != len(feature_names):
        raise ValueError("Overall accuracies must match the feature count.")
    if len(annotated_accuracies) != len(feature_names):
        raise ValueError("Annotated accuracies must match the feature count.")
    if any(not name or name.strip() != name for name in feature_names):
        raise ValueError("Morphology feature names must be non-empty and trimmed.")

    feature_name_width = max(len(name) for name in feature_names)

    return tuple(
        f"{prefix} {feature_name:<{feature_name_width}}    "
        f"overall={overall:.6f}    annotated={annotated:.6f}"
        for feature_name, overall, annotated in zip(
            feature_names,
            overall_accuracies,
            annotated_accuracies,
            strict=True,
        )
    )


def format_token_slice_metric_rows(
    *,
    slice_name: str,
    metrics: TokenTaskEvaluationMetrics,
    universal_dependencies: UniversalDependenciesEvaluationMetrics | None = None,
) -> tuple[str, ...]:
    if not slice_name or slice_name.strip() != slice_name:
        raise ValueError("Token-slice name must be non-empty and trimmed.")

    morphology_metrics = metrics.morphology_micro_metrics()
    universal_dependencies_values = (
        ()
        if universal_dependencies is None
        else (
            ("UD UPOS F1", f"{universal_dependencies.upos.f1:.6f}"),
            ("UD UFeats F1", f"{universal_dependencies.ufeats.f1:.6f}"),
            ("UD Lemmas F1", f"{universal_dependencies.lemmas.f1:.6f}"),
        )
    )
    metric_values = (
        ("tokens", str(metrics.token_count)),
        ("UPOS accuracy", f"{metrics.upos_accuracy:.6f}"),
        *universal_dependencies_values,
        ("lemma annotations", str(metrics.lemma_annotation_count)),
        ("lemma-rule targets", str(metrics.lemma_target_count)),
        (
            "lemma-rule coverage",
            "undefined"
            if metrics.lemma_rule_coverage is None
            else f"{metrics.lemma_rule_coverage:.6f}",
        ),
        (
            "lemma-rule accuracy",
            "undefined"
            if metrics.lemma_rule_accuracy is None
            else f"{metrics.lemma_rule_accuracy:.6f}",
        ),
        (
            "lemma end-to-end accuracy",
            "undefined"
            if metrics.lemma_end_to_end_accuracy is None
            else f"{metrics.lemma_end_to_end_accuracy:.6f}",
        ),
        ("morphology micro precision", f"{morphology_metrics.precision:.6f}"),
        ("morphology micro recall", f"{morphology_metrics.recall:.6f}"),
        ("morphology micro F1", f"{morphology_metrics.f1:.6f}"),
    )
    label_width = max(len(label) for label, _ in metric_values)

    return tuple(
        f"{slice_name} {label:<{label_width}}    {value}"
        for label, value in metric_values
    )


def format_morphology_error_attribution_rows(
    *,
    slice_name: str,
    feature_names: Sequence[str],
    metrics: TokenTaskEvaluationMetrics,
) -> tuple[str, ...]:
    if not slice_name or slice_name.strip() != slice_name:
        raise ValueError("Token-slice name must be non-empty and trimmed.")
    if not feature_names:
        raise ValueError("Morphology attribution must contain features.")
    if any(not name or name.strip() != name for name in feature_names):
        raise ValueError("Morphology feature names must be non-empty and trimmed.")
    if len(metrics.morphology_correct_counts) != len(feature_names):
        raise ValueError("Morphology correct counts must match the feature count.")
    if any(
        correct_count < 0 or correct_count > metrics.token_count
        for correct_count in metrics.morphology_correct_counts
    ):
        raise ValueError("Morphology correct counts must be valid token counts.")

    feature_errors = tuple(
        (
            feature_name,
            metrics.token_count - correct_count,
            correct_count / metrics.token_count,
        )
        for feature_name, correct_count in zip(
            feature_names,
            metrics.morphology_correct_counts,
            strict=True,
        )
    )
    total_error_count = sum(error_count for _, error_count, _ in feature_errors)
    if total_error_count == 0:
        return (f"{slice_name} no morphology feature errors",)

    ranked_errors = tuple(
        sorted(
            (feature_error for feature_error in feature_errors if feature_error[1] > 0),
            key=lambda feature_error: (-feature_error[1], feature_error[0]),
        )
    )
    feature_name_width = max(len(name) for name, _, _ in ranked_errors)
    error_count_width = max(
        len(str(error_count)) for _, error_count, _ in ranked_errors
    )

    return tuple(
        f"{slice_name} {feature_name:<{feature_name_width}}    "
        f"errors={error_count:>{error_count_width}}    "
        f"share={error_count / total_error_count:>7.2%}    "
        f"accuracy={accuracy:.6f}"
        for feature_name, error_count, accuracy in ranked_errors
    )
