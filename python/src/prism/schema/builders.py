from collections.abc import Iterable, Mapping

from prism.schema.morphology import (
    MORPHOLOGY_SCHEMA_VERSION,
    MorphologyFeatureSchema,
    MorphologySchema,
)

from prism.schema.lemma import (
    LEMMA_RULE_SCHEMA_VERSION,
    LemmaRuleSchema,
    derive_lemma_edit_rule,
)

from prism.schema.upos import (
    UPOS_SCHEMA_VERSION,
    UposSchema,
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
            allows_multiple_values=(feature_name in multiple_value_features),
        )
        for feature_name, values in sorted(values_by_feature.items())
    )

    return MorphologySchema(
        version=MORPHOLOGY_SCHEMA_VERSION,
        features=features,
    )


def build_lemma_rule_schema(
    token_lemma_pairs: Iterable[tuple[str, str]],
) -> LemmaRuleSchema:
    rules = {derive_lemma_edit_rule(token, lemma) for token, lemma in token_lemma_pairs}

    return LemmaRuleSchema(
        version=LEMMA_RULE_SCHEMA_VERSION,
        rules=tuple(
            sorted(
                rules,
                key=lambda rule: rule.sort_key,
            )
        ),
    )


def build_upos_schema(
    labels: Iterable[str],
) -> UposSchema:
    return UposSchema(version=UPOS_SCHEMA_VERSION, labels=tuple(sorted(set(labels))))
