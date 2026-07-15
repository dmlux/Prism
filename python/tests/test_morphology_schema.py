import pytest

from prism.schema import (
    MORPHOLOGY_SCHEMA_VERSION,
    MorphologyFeatureSchema,
    MorphologySchema,
)


def test_morphology_feature_schema_exposes_deterministic_labels() -> None:
    schema = MorphologyFeatureSchema(
        name="Number",
        values=("Plur", "Sing"),
        allows_multiple_values=True,
    )

    assert schema.name == "Number"
    assert schema.values == ("Plur", "Sing")
    assert schema.allows_multiple_values is True
    assert schema.labels == ("<NONE>", "Plur", "Sing")


@pytest.mark.parametrize(
    ("name", "values", "expected_message"),
    [
        ("", ("Sing",), "name must be non-empty"),
        (" Number", ("Sing",), "name must be non-empty"),
        ("Number", (), "values must not be empty"),
        ("Number", ("<NONE>", "Sing"), "<NONE> is reserved"),
        ("Number", ("Sing", "Sing"), "values must be unique"),
        ("Number", ("Sing", "Plur"), "values must be sorted"),
    ],
)
def test_morphology_feature_schema_rejects_invalid_data(
    name: str,
    values: tuple[str, ...],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        MorphologyFeatureSchema(
            name=name,
            values=values,
            allows_multiple_values=False,
        )


def test_morphology_schema_preserves_versioned_feature_order() -> None:
    gender = MorphologyFeatureSchema(
        name="Gender",
        values=("Fem", "Masc", "Neut"),
        allows_multiple_values=True,
    )
    number = MorphologyFeatureSchema(
        name="Number",
        values=("Plur", "Sing"),
        allows_multiple_values=True,
    )

    schema = MorphologySchema(
        version=MORPHOLOGY_SCHEMA_VERSION,
        features=(gender, number),
    )

    assert schema.version == 1
    assert schema.features == (gender, number)


@pytest.mark.parametrize(
    ("version", "feature_names", "expected_message"),
    [
        (0, ("Number",), "version must be positive"),
        (1, (), "features must not be empty"),
        (1, ("Number", "Number"), "names must be unique"),
        (1, ("Number", "Gender"), "sorted by name"),
    ],
)
def test_morphology_schema_rejects_invalid_structure(
    version: int,
    feature_names: tuple[str, ...],
    expected_message: str,
) -> None:
    features = tuple(
        MorphologyFeatureSchema(
            name=name,
            values=("Value",),
            allows_multiple_values=False,
        )
        for name in feature_names
    )

    with pytest.raises(ValueError, match=expected_message):
        MorphologySchema(
            version=version,
            features=features,
        )
