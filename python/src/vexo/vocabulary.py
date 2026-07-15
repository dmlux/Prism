from collections import Counter

from vexo.conllu import Token

PAD_TOKEN = "<PAD>"
PAD_CHARACTER = "<PAD_CHAR>"
UNKNOWN_TOKEN = "<UNK>"
UNKNOWN_CHARACTER = "<UNK_CHAR>"
NO_FEATURE = "<NONE>"

def build_word_vocabulary(
    sentences: list[list[Token]],
    minimum_frequency: int = 2,
) -> dict[str, int]:
    counts = Counter(
        token.text
        for sentence in sentences
        for token in sentence
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNKNOWN_TOKEN: 1,
    }

    for text in sorted(counts):
        if counts[text] >= minimum_frequency:
            vocabulary[text] = len(vocabulary)

    return vocabulary

def build_tag_vocabulary(
    sentences: list[list[Token]],
) -> dict[str, int]:
    tags = sorted(
        {
            token.upos
            for sentence in sentences
            for token in sentence
        }
    )

    return {
        tag: index
        for index, tag in enumerate(tags)
    }

def encode_sentence(
    sentence: list[Token],
    word_vocabulary: dict[str, int],
    tag_vocabulary: dict[str, int],
) -> tuple[list[int], list[int]]:
    word_ids = [
        word_vocabulary.get(token.text, word_vocabulary[UNKNOWN_TOKEN])
        for token in sentence
    ]
    tag_ids = [
        tag_vocabulary[token.upos]
        for token in sentence
    ]

    return word_ids, tag_ids

def build_character_vocabulary(
    sentences: list[list[Token]],
) -> dict[str, int]:
    characters = sorted(
        {
            character
            for sentence in sentences
            for token in sentence
            for character in token.text
        }
    )

    vocabulary = {
        PAD_CHARACTER: 0,
        UNKNOWN_CHARACTER: 1,
    }

    for character in characters:
        vocabulary[character] = len(vocabulary)

    return vocabulary

def encode_sentence_characters(
    sentence: list[Token],
    character_vocabulary: dict[str, int],
) -> list[list[int]]:
    unknown_id = character_vocabulary[UNKNOWN_CHARACTER]

    return [
        [
            character_vocabulary.get(character, unknown_id)
            for character in token.text
        ]
        for token in sentence
    ]

def build_feature_vocabulary(
    sentences: list[list[Token]],
    feature_name: str,
) -> dict[str, int]:
    values = sorted(
        {
            token.features[feature_name]
            for sentence in sentences
            for token in sentence
            if feature_name in token.features
        }
    )

    vocabulary = {
        NO_FEATURE: 0,
    }

    for value in values:
        vocabulary[value] = len(vocabulary)

    return vocabulary

def encode_sentence_feature(
    sentence: list[Token],
    feature_name: str,
    feature_vocabulary: dict[str, int],
) -> list[int]:
    return [
        feature_vocabulary[
            token.features.get(feature_name, NO_FEATURE)
        ]
        for token in sentence
    ]