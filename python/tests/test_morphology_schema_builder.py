from prism.schema import (
    MORPHOLOGY_SCHEMA_VERSION,
    MorphologyFeatureSchema,
    build_morphology_schema,
)


def test_build_morphology_schema_collects_sorted_atomic_values() -> None:
    feature_maps = [
        {
            "Number": "Sing",
            "Gender": "Masc",
        },
        {
            "Number": "Plur",
            "Gender": "Fem,Masc",
        },
        {},
    ]

    schema = build_morphology_schema(feature_maps)

    assert schema.version == MORPHOLOGY_SCHEMA_VERSION
    assert schema.features == (
        MorphologyFeatureSchema(
            name="Gender",
            values=("Fem", "Masc"),
            allows_multiple_values=True,
        ),
        MorphologyFeatureSchema(
            name="Number",
            values=("Plur", "Sing"),
            allows_multiple_values=False,
        ),
    )
