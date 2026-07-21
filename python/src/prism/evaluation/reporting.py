from collections.abc import Sequence

from prism.evaluation.classification import ClassificationMetrics


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
