import pytest

from prism.schema import (
    MorphologyFeatureSchema,
    MorphologySchema,
    encode_morphology_targets,
    decode_morphology_values,
)


def test_encode_morphology_targets_uses_none_and_atomic_values() -> None:
    schema = MorphologySchema(
        version=1,
        features=(
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
        ),
    )

    targets = encode_morphology_targets(
        schema,
        {
            "Gender": "Fem,Masc",
        },
    )

    assert targets == (
        (False, True, True),
        (True, False, False),
    )


@pytest.mark.parametrize(
    ("features", "expected_message"),
    [
        (
            {"Case": "Nom"},
            "Unknown morphology feature: Case",
        ),
        (
            {"Number": "Dual"},
            "Unknown value 'Dual' for morphology feature 'Number'",
        ),
        (
            {"Number": "Plur,Sing"},
            "does not allow multiple values",
        ),
    ],
)
def test_encode_morphology_targets_rejects_invalid_annotations(
    features: dict[str, str],
    expected_message: str,
) -> None:
    schema = MorphologySchema(
        version=1,
        features=(
            MorphologyFeatureSchema(
                name="Number",
                values=("Plur", "Sing"),
                allows_multiple_values=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match=expected_message):
        encode_morphology_targets(schema, features)


def test_decode_morphology_values_restores_ud_annotations() -> None:
    schema = MorphologySchema(
        version=1,
        features=(
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
        ),
    )

    features = decode_morphology_values(
        schema,
        (
            (False, True, True),
            (True, False, False),
        ),
    )

    assert features == {
        "Gender": "Fem,Masc",
    }


@pytest.mark.parametrize(
    ("encoded_features", "expected_message"),
    [
        (
            (),
            "feature count does not match schema",
        ),
        (
            ((False, False, False),),
            "must activate at least one label",
        ),
        (
            ((True, True, False),),
            "<NONE> cannot be combined",
        ),
        (
            ((False, True, True),),
            "does not allow multiple values",
        ),
        (
            ((True, False),),
            "label count does not match schema",
        ),
    ],
)
def test_decode_morphology_values_rejects_invalid_predictions(
    encoded_features: tuple[tuple[bool, ...], ...],
    expected_message: str,
) -> None:
    schema = MorphologySchema(
        version=1,
        features=(
            MorphologyFeatureSchema(
                name="Number",
                values=("Plur", "Sing"),
                allows_multiple_values=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match=expected_message):
        decode_morphology_values(schema, encoded_features)