from prism.conllu import Token
from prism.vocabulary import (
    NO_FEATURE,
    PAD_CHARACTER,
    UNKNOWN_CHARACTER,
    build_character_vocabulary,
    build_feature_vocabulary,
    encode_sentence_characters,
    encode_sentence_feature,
)

def make_token(
    text: str,
    features: dict[str, str] | None = None,
) -> Token:
    return Token(
        text=text,
        lemma=text,
        upos="X",
        features=features or {},
    )

def test_character_vocabulary_and_encoding() -> None:
    sentence = [
        make_token("på"),
        make_token("bok"),
    ]

    vocabulary = build_character_vocabulary([sentence])

    assert vocabulary[PAD_CHARACTER] == 0
    assert vocabulary[UNKNOWN_CHARACTER] == 1

    encoded = encode_sentence_characters(
        [make_token("på!")],
        vocabulary,
    )

    assert encoded == [[
        vocabulary["p"],
        vocabulary["å"],
        vocabulary[UNKNOWN_CHARACTER],
    ]]

def test_feature_vocabulary() -> None:
    sentence = [
        make_token("bok", {"Number": "Sing"}),
        make_token("bøker", {"Number": "Plur"}),
        make_token("og"),
    ]

    vocabulary = build_feature_vocabulary(
        [sentence],
        "Number",
    )

    assert vocabulary == {
        NO_FEATURE: 0,
        "Plur": 1,
        "Sing": 2,
    }

    encoded = encode_sentence_feature(
        sentence,
        "Number",
        vocabulary,
    )

    assert encoded == [
        vocabulary["Sing"],
        vocabulary["Plur"],
        vocabulary[NO_FEATURE],
    ]