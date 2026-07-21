import pytest

from prism.schema import (
    CHARACTER_FIRST_LITERAL_ID,
    CHARACTER_UNKNOWN_ID,
    build_character_vocabulary_schema,
)
from prism.schema.serialization import (
    deserialize_character_vocabulary_schema,
    serialize_character_vocabulary_schema,
)


def test_character_vocabulary_is_normalized_and_deterministic() -> None:
    vocabulary = build_character_vocabulary_schema(
        tokens=("så", "sa\u030a", "øl"),
    )

    assert vocabulary.characters == tuple(sorted(set("såøl")))
    assert vocabulary.size == CHARACTER_FIRST_LITERAL_ID + len(vocabulary.characters)
    assert vocabulary.character_id("å") >= CHARACTER_FIRST_LITERAL_ID
    assert vocabulary.character_id("x") == CHARACTER_UNKNOWN_ID


def test_character_vocabulary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires tokens"):
        build_character_vocabulary_schema(tokens=())


def test_character_vocabulary_serialization_round_trip() -> None:
    vocabulary = build_character_vocabulary_schema(tokens=("Bokmål", "Nynorsk"))

    restored = deserialize_character_vocabulary_schema(
        serialize_character_vocabulary_schema(vocabulary)
    )

    assert restored == vocabulary
