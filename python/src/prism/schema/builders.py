from collections.abc import Iterable, Mapping

from prism.schema.morphology import (
    MORPHOLOGY_SCHEMA_VERSION,
    MorphologyFeatureSchema,
    MorphologySchema,
)


def build_morphology_schema(
    feature_maps: Iterable[Mapping[str, str]],
) -> MorphologySchema:
    values_by_feature: dict[str, set[str]] = {}
    multiple_value_features: set[str] = set()

    for feature_map in feature_maps:
        for feature_name, raw_value in feature_map.items():
            values = tuple(raw_value.split(","))

            values_by_feature.setdefault(
                feature_name,
                set(),
            ).update(values)

            if len(values) > 1:
                multiple_value_features.add(feature_name)

    features = tuple(
        MorphologyFeatureSchema(
            name=feature_name,
            values=tuple(sorted(values)),
            allows_multiple_values=(
                feature_name in multiple_value_features
            ),
        )
        for feature_name, values in sorted(values_by_feature.items())
    )

    return MorphologySchema(
        version=MORPHOLOGY_SCHEMA_VERSION,
        features=features,
    )
