from collections.abc import Mapping

from prism.schema.characters import CharacterVocabularySchema
from prism.schema.token_tasks import TokenTaskSchema


TOKEN_TASK_SCHEMA_FORMAT_VERSION = 1


def serialize_character_vocabulary_schema(
    schema: CharacterVocabularySchema,
) -> dict[str, object]:
    return {
        "version": schema.version,
        "characters": list(schema.characters),
    }


def deserialize_character_vocabulary_schema(
    value: object,
) -> CharacterVocabularySchema:
    if not isinstance(value, Mapping):
        raise ValueError("Character vocabulary metadata must be a mapping.")

    version = value.get("version")
    characters = value.get("characters")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("Character vocabulary version must be an integer.")
    if not isinstance(characters, (list, tuple)) or not all(
        isinstance(character, str) for character in characters
    ):
        raise ValueError("Character vocabulary characters must be strings.")

    return CharacterVocabularySchema(
        version=version,
        characters=tuple(characters),
    )


def serialize_token_task_schema(schema: TokenTaskSchema) -> dict[str, object]:
    return {
        "format_version": TOKEN_TASK_SCHEMA_FORMAT_VERSION,
        "upos": {
            "version": schema.upos.version,
            "labels": list(schema.upos.labels),
        },
        "morphology": {
            "version": schema.morphology.version,
            "features": [
                {
                    "name": feature.name,
                    "values": list(feature.values),
                    "allows_multiple_values": (feature.allows_multiple_values),
                }
                for feature in schema.morphology.features
            ],
        },
        "lemma_rules": {
            "version": schema.lemma_rules.version,
            "rules": [
                {
                    "prefix_removal": rule.prefix_removal,
                    "suffix_removal": rule.suffix_removal,
                    "prefix_addition": rule.prefix_addition,
                    "suffix_addition": rule.suffix_addition,
                }
                for rule in schema.lemma_rules.rules
            ],
        },
    }
