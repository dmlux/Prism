from prism.conllu import Token
from prism.data.norwegian_bokmaal import (
    build_norwegian_bokmaal_schema,
)
from prism.schema import LemmaEditRule, MorphologyFeatureSchema


def test_build_norwegian_bokmaal_schema_uses_training_data() -> None:
    schema = build_norwegian_bokmaal_schema(
        [
            [
                Token(
                    text="husene", lemma="hus", upos="NOUN", features={"Number": "Plur"}
                ),
                Token(text=".", lemma="$.", upos="PUNCT", features={}),
            ]
        ]
    )

    assert schema.upos.labels == (
        "NOUN",
        "PUNCT",
    )
    assert schema.morphology.features == (
        MorphologyFeatureSchema(
            name="Number",
            values=("Plur",),
            allows_multiple_values=False,
        ),
    )
    assert schema.lemma_rules.rules == (
        LemmaEditRule(
            prefix_removal=0,
            suffix_removal=0,
            prefix_addition="",
            suffix_addition="",
        ),
        LemmaEditRule(
            prefix_removal=0,
            suffix_removal=3,
            prefix_addition="",
            suffix_addition="",
        ),
    )
