import torch

from prism.modeling import encode_character_token_batch
from prism.schema import (
    CHARACTER_END_ID,
    CHARACTER_START_ID,
    CHARACTER_TRUNCATION_ID,
    CHARACTER_UNKNOWN_ID,
    build_character_vocabulary_schema,
)


def test_character_batch_preserves_boundaries_unknowns_and_padding() -> None:
    vocabulary = build_character_vocabulary_schema(tokens=("katt", "hund"))

    batch = encode_character_token_batch(
        token_sequences=(("katt", "x"), ("hund",)),
        vocabulary=vocabulary,
        maximum_character_count=8,
    )

    assert batch.character_ids.shape == (2, 2, 8)
    assert batch.token_mask.tolist() == [[True, True], [True, False]]
    assert batch.character_ids[0, 0, 0].item() == CHARACTER_START_ID
    assert batch.character_ids[0, 0, 5].item() == CHARACTER_END_ID
    assert batch.character_ids[0, 1, 1].item() == CHARACTER_UNKNOWN_ID
    assert not batch.character_mask[1, 1].any().item()
    assert torch.equal(batch.to(torch.device("cpu")).token_mask, batch.token_mask)


def test_character_batch_retains_prefix_and_suffix_when_truncated() -> None:
    vocabulary = build_character_vocabulary_schema(tokens=("abcdefghij",))

    batch = encode_character_token_batch(
        token_sequences=(("abcdefghij",),),
        vocabulary=vocabulary,
        maximum_character_count=8,
    )

    encoded = batch.character_ids[0, 0].tolist()
    assert encoded[0] == CHARACTER_START_ID
    assert encoded[-1] == CHARACTER_END_ID
    assert encoded[4] == CHARACTER_TRUNCATION_ID
    assert batch.character_mask[0, 0].all().item()
