from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import unicodedata


CHARACTER_VOCABULARY_SCHEMA_VERSION = 1
CHARACTER_PADDING_ID = 0
CHARACTER_UNKNOWN_ID = 1
CHARACTER_START_ID = 2
CHARACTER_END_ID = 3
CHARACTER_TRUNCATION_ID = 4
CHARACTER_FIRST_LITERAL_ID = 5


def normalize_character_token(token: str) -> str:
    if not token:
        raise ValueError("Character input token must not be empty.")

    return unicodedata.normalize("NFC", token)


@dataclass(frozen=True, slots=True, kw_only=True)
class CharacterVocabularySchema:
    version: int
    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("Character vocabulary version must be positive.")
        if not self.characters:
            raise ValueError("Character vocabulary must contain literal characters.")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Character vocabulary must not contain duplicates.")
        if any(len(character) != 1 for character in self.characters):
            raise ValueError(
                "Character vocabulary entries must be Unicode code points."
            )
        if any(
            unicodedata.normalize("NFC", character) != character
            for character in self.characters
        ):
            raise ValueError("Character vocabulary entries must use Unicode NFC.")

    @property
    def size(self) -> int:
        return CHARACTER_FIRST_LITERAL_ID + len(self.characters)

    def character_id(self, character: str) -> int:
        if len(character) != 1:
            raise ValueError("Character lookup requires one Unicode code point.")

        try:
            index = self.characters.index(character)
        except ValueError:
            return CHARACTER_UNKNOWN_ID

        return CHARACTER_FIRST_LITERAL_ID + index


def build_character_vocabulary_schema(
    *,
    tokens: Iterable[str],
    minimum_frequency: int = 1,
) -> CharacterVocabularySchema:
    if minimum_frequency <= 0:
        raise ValueError("Minimum character frequency must be positive.")

    frequencies: Counter[str] = Counter()
    token_count = 0
    for token in tokens:
        frequencies.update(normalize_character_token(token))
        token_count += 1

    if token_count == 0:
        raise ValueError("Character vocabulary requires tokens.")

    characters = tuple(
        sorted(
            character
            for character, frequency in frequencies.items()
            if frequency >= minimum_frequency
        )
    )
    if not characters:
        raise ValueError("Character frequency threshold removed every character.")

    return CharacterVocabularySchema(
        version=CHARACTER_VOCABULARY_SCHEMA_VERSION,
        characters=characters,
    )
