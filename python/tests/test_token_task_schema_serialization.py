import json

from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)
from prism.schema.serialization import (
    serialize_token_task_schema,
)


def test_serialize_token_task_schema_produces_json_data() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(
            version=1,
            labels=("NOUN", "VERB"),
        ),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Case",
                    values=("Acc", "Nom"),
                    allows_multiple_values=False,
                ),
            ),
        ),
        lemma_rules=LemmaRuleSchema(
            version=1,
            rules=(
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=1,
                    prefix_addition="",
                    suffix_addition="e",
                ),
            ),
        ),
    )

    serialized = serialize_token_task_schema(schema)

    assert serialized == {
        "format_version": 1,
        "upos": {
            "version": 1,
            "labels": ["NOUN", "VERB"],
        },
        "morphology": {
            "version": 1,
            "features": [
                {
                    "name": "Case",
                    "values": ["Acc", "Nom"],
                    "allows_multiple_values": False,
                }
            ],
        },
        "lemma_rules": {
            "version": 1,
            "rules": [
                {
                    "prefix_removal": 0,
                    "suffix_removal": 1,
                    "prefix_addition": "",
                    "suffix_addition": "e",
                }
            ],
        },
    }

    assert json.loads(json.dumps(serialized)) == serialized
