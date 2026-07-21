import torch

from prism.modeling import CharacterCnnTokenEncoder, encode_character_token_batch
from prism.schema import build_character_vocabulary_schema


def test_character_cnn_encodes_tokens_and_zeros_padding() -> None:
    vocabulary = build_character_vocabulary_schema(tokens=("katt", "hund"))
    batch = encode_character_token_batch(
        token_sequences=(("katt", "hund"), ("hund",)),
        vocabulary=vocabulary,
        maximum_character_count=8,
    )
    encoder = CharacterCnnTokenEncoder(
        vocabulary_size=vocabulary.size,
        hidden_size=12,
        embedding_size=4,
    )

    hidden_states = encoder(batch)
    hidden_states.sum().backward()

    assert hidden_states.shape == (2, 2, 12)
    torch.testing.assert_close(hidden_states[1, 1], torch.zeros(12))
    assert encoder.embedding.weight.grad is not None
