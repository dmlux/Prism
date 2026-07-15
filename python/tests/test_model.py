import torch

from prism.model import (
    CharacterBiLSTMMultiTaskTagger,
    CharacterBiLSTMPosTagger,
    CharacterEncoder,
)


def test_character_encoder_shape_and_padding() -> None:
    character_ids = torch.tensor([
        [
            [2, 3, 0],
            [4, 5, 6],
        ],
        [
            [7, 8, 0],
            [0, 0, 0],
        ],
    ])
    character_lengths = torch.tensor([
        [2, 3],
        [2, 0],
    ])

    encoder = CharacterEncoder(
        character_count=9,
        embedding_size=4,
        hidden_size=3,
    )

    representation = encoder(
        character_ids,
        character_lengths,
    )

    assert representation.shape == (2, 2, 6)
    assert torch.equal(
        representation[1, 1],
        torch.zeros(6),
    )

def test_character_pos_tagger_output_shape() -> None:
    word_ids = torch.tensor([
        [2, 3],
        [4, 0],
    ])
    character_ids = torch.tensor([
        [
            [2, 3, 0],
            [4, 5, 6],
        ],
        [
            [7, 8, 0],
            [0, 0, 0],
        ],
    ])
    sentence_lengths = torch.tensor([2, 1])
    character_lengths = torch.tensor([
        [2, 3],
        [2, 0],
    ])

    model = CharacterBiLSTMPosTagger(
        vocabulary_size=5,
        character_count=9,
        tag_count=3,
        word_embedding_size=4,
        character_embedding_size=4,
        character_hidden_size=3,
        hidden_size=5,
    )

    outputs = model(
        word_ids,
        character_ids,
        sentence_lengths,
        character_lengths,
    )

    assert outputs.shape == (2, 2, 3)

def test_multi_task_tagger_output_shapes() -> None:
    word_ids = torch.tensor([
        [2, 3],
        [4, 0],
    ])
    character_ids = torch.tensor([
        [
            [2, 3, 0],
            [4, 5, 6],
        ],
        [
            [7, 8, 0],
            [0, 0, 0],
        ],
    ])
    sentence_lengths = torch.tensor([2, 1])
    character_lengths = torch.tensor([
        [2, 3],
        [2, 0],
    ])

    model = CharacterBiLSTMMultiTaskTagger(
        vocabulary_size=5,
        character_count=9,
        tag_count=3,
        feature_count=4,
        word_embedding_size=4,
        character_embedding_size=4,
        character_hidden_size=3,
        hidden_size=5,
    )

    pos_outputs, number_outputs = model(
        word_ids,
        character_ids,
        sentence_lengths,
        character_lengths,
    )

    assert pos_outputs.shape == (2, 2, 3)
    assert number_outputs.shape == (2, 2, 4)