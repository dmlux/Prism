from vexo.conllu import Token
from vexo.vocabulary import (
    PAD_CHARACTER,
    UNKNOWN_CHARACTER,
    build_character_vocabulary,
    encode_sentence_characters,
)

def make_token(text: str) -> Token:
    return Token(
        text=text,
        lemma=text,
        upos="X",
        features={},
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