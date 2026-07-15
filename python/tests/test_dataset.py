from vexo.conllu import Token
from vexo.dataset import (
    CharacterPosDataset,
    collate_character_sentences,
)
from vexo.vocabulary import (
    build_character_vocabulary,
    build_tag_vocabulary,
    build_word_vocabulary,
)

def make_token(text: str, upos: str) -> Token:
    return Token(
        text=text,
        lemma=text,
        upos=upos,
        features={},
    )

def test_character_batch_padding() -> None:
    sentences = [
        [
            make_token("på", "ADP"),
            make_token("bok", "NOUN"),
        ],
        [
            make_token("hei", "INTJ"),
        ],
    ]

    words = build_word_vocabulary(
        sentences,
        minimum_frequency=1,
    )
    tags = build_tag_vocabulary(sentences)
    characters = build_character_vocabulary(sentences)

    dataset = CharacterPosDataset(
        sentences,
        words,
        tags,
        characters,
    )

    (
        word_ids,
        character_ids,
        tag_ids,
        sentence_lengths,
        character_lengths,
    ) = collate_character_sentences([
        dataset[0],
        dataset[1],
    ])

    assert word_ids.shape == (2, 2)
    assert character_ids.shape == (2, 2, 3)
    assert tag_ids.shape == (2, 2)

    assert sentence_lengths.tolist() == [2, 1]
    assert character_lengths.tolist() == [
        [2, 3],
        [3, 0],
    ]

    assert word_ids[1, 1].item() == 0
    assert tag_ids[1, 1].item() == -100