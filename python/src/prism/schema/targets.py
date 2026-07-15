from collections.abc import Mapping, Sequence

from prism.schema.morphology import (
    NO_MORPHOLOGY_VALUE,
    MorphologySchema,
)


def encode_morphology_targets(
    schema: MorphologySchema,
    features: Mapping[str, str],
) -> tuple[tuple[bool, ...], ...]:
    schema_feature_names = {
        feature_schema.name
        for feature_schema in schema.features
    }
    unknown_feature_names = sorted(
        set(features).difference(schema_feature_names)
    )

    if unknown_feature_names:
        raise ValueError(
            f"Unknown morphology feature: {unknown_feature_names[0]}"
        )

    encoded_features: list[tuple[bool, ...]] = []

    for feature_schema in schema.features:
        raw_value = features.get(feature_schema.name)

        if raw_value is None:
            active_values = {NO_MORPHOLOGY_VALUE}
        else:
            values = tuple(raw_value.split(","))

            if (
                len(values) > 1
                and not feature_schema.allows_multiple_values
            ):
                raise ValueError(
                    f"Morphology feature {feature_schema.name!r} "
                    "does not allow multiple values."
                )

            unknown_values = sorted(
                set(values).difference(feature_schema.values)
            )
            if unknown_values:
                raise ValueError(
                    f"Unknown value {unknown_values[0]!r} "
                    f"for morphology feature {feature_schema.name!r}."
                )

            active_values = set(values)

        encoded_features.append(
            tuple(
                label in active_values
                for label in feature_schema.labels
            )
        )

    return tuple(encoded_features)


def decode_morphology_values(
    schema: MorphologySchema,
    encoded_features: Sequence[Sequence[bool]],
) -> dict[str, str]:
    if len(encoded_features) != len(schema.features):
        raise ValueError(
            "Encoded morphology feature count does not match schema."
        )

    decoded_features: dict[str, str] = {}

    for feature_schema, encoded_values in zip(
        schema.features,
        encoded_features,
        strict=True,
    ):
        if len(encoded_values) != len(feature_schema.labels):
            raise ValueError(
                "Encoded label count does not match schema "
                f"for morphology feature {feature_schema.name!r}."
            )

        active_values = tuple(
            label
            for label, is_active in zip(
                feature_schema.labels,
                encoded_values,
                strict=True,
            )
            if is_active
        )

        if not active_values:
            raise ValueError(
                f"Morphology feature {feature_schema.name!r} "
                "must activate at least one label."
            )

        if NO_MORPHOLOGY_VALUE in active_values:
            if len(active_values) > 1:
                raise ValueError(
                    f"{NO_MORPHOLOGY_VALUE} cannot be combined "
                    "with morphology values."
                )

            continue

        if (
            len(active_values) > 1
            and not feature_schema.allows_multiple_values
        ):
            raise ValueError(
                f"Morphology feature {feature_schema.name!r} "
                "does not allow multiple values."
            )

        decoded_features[feature_schema.name] = ",".join(
            active_values
        )

    return decoded_features