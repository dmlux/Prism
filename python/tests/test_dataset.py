from prism.conllu import Token
from prism.baselines.recurrent.dataset import (
    CharacterFeatureDataset,
    CharacterPosDataset,
    collate_character_sentences,
    collate_character_feature_sentences,
)
from prism.baselines.recurrent.vocabulary import (
    NO_FEATURE,
    build_character_vocabulary,
    build_feature_vocabulary,
    build_tag_vocabulary,
    build_word_vocabulary,
)


def make_token(
    text: str,
    upos: str,
    features: dict[str, str] | None = None,
) -> Token:
    return Token(
        text=text,
        lemma=text,
        upos=upos,
        features=features or {},
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
    ) = collate_character_sentences(
        [
            dataset[0],
            dataset[1],
        ]
    )

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


def test_character_feature_dataset() -> None:
    sentences = [
        [
            make_token(
                "bok",
                "Noun",
                {"Number": "Sing"},
            ),
            make_token("og", "CCONJ"),
        ],
        [
            make_token(
                "bøker",
                "NOUN",
                {"Number": "Plur"},
            )
        ],
    ]

    words = build_word_vocabulary(
        sentences,
        minimum_frequency=1,
    )
    tags = build_tag_vocabulary(sentences)
    characters = build_character_vocabulary(sentences)
    numbers = build_feature_vocabulary(
        sentences,
        "Number",
    )

    dataset = CharacterFeatureDataset(
        sentences,
        words,
        tags,
        characters,
        "Number",
        numbers,
    )

    assert len(dataset) == 2

    _, _, number_ids, _ = dataset[0]

    assert number_ids.tolist() == [
        numbers["Sing"],
        numbers[NO_FEATURE],
    ]

    (
        _,
        _,
        _,
        number_batch,
        sentence_lengths,
        _,
    ) = collate_character_feature_sentences(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert number_batch.shape == (2, 2)
    assert number_batch[1].tolist() == [
        numbers["Plur"],
        -100,
    ]
    assert sentence_lengths.tolist() == [2, 1]
