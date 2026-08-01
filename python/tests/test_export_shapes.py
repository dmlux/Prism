import pytest
import torch

from prism.exporting import (
    FixedExportShapes,
    pad_character_token_batch,
    pad_tokenized_batch,
    repeat_pad_sentences,
)
from prism.modeling import CharacterTokenBatch, TokenizedBatch


def _tokenized_batch() -> TokenizedBatch:
    return TokenizedBatch(
        input_ids=torch.tensor([[101, 11, 12, 102]], dtype=torch.long),
        attention_mask=torch.tensor([[True, True, True, True]]),
        first_subword_indices=torch.tensor([[1, 2]], dtype=torch.long),
        subword_end_indices=torch.tensor([[2, 3]], dtype=torch.long),
        token_mask=torch.tensor([[True, True]]),
    )


def _character_batch() -> CharacterTokenBatch:
    return CharacterTokenBatch(
        character_ids=torch.tensor([[[2, 5, 3, 0, 0], [2, 6, 3, 0, 0]]]),
        character_mask=torch.tensor(
            [[[True, True, True, False, False], [True, True, True, False, False]]]
        ),
        token_mask=torch.tensor([[True, True]]),
    )


def _shapes(character_count: int | None = 5) -> FixedExportShapes:
    return FixedExportShapes(
        batch_size=1,
        subword_count=6,
        token_count=4,
        character_count=character_count,
    )


def test_fixed_export_shapes_validates_sizes() -> None:
    with pytest.raises(ValueError):
        FixedExportShapes(
            batch_size=0,
            subword_count=6,
            token_count=4,
            character_count=None,
        )
    with pytest.raises(ValueError):
        FixedExportShapes(
            batch_size=1,
            subword_count=6,
            token_count=4,
            character_count=4,
        )


def test_repeat_pad_sentences_fills_partial_batches() -> None:
    assert repeat_pad_sentences(("a", "b"), batch_size=4) == ("a", "b", "b", "b")
    assert repeat_pad_sentences(("a",), batch_size=1) == ("a",)

    with pytest.raises(ValueError):
        repeat_pad_sentences((), batch_size=2)
    with pytest.raises(ValueError):
        repeat_pad_sentences(("a", "b", "c"), batch_size=2)


def test_pad_tokenized_batch_pads_to_fixed_shapes() -> None:
    padded = pad_tokenized_batch(
        _tokenized_batch(),
        shapes=_shapes(),
        padding_token_id=7,
    )

    assert padded.input_ids.tolist() == [[101, 11, 12, 102, 7, 7]]
    assert padded.attention_mask.tolist() == [[True, True, True, True, False, False]]
    assert padded.first_subword_indices.tolist() == [[1, 2, 0, 0]]
    assert padded.subword_end_indices.tolist() == [[2, 3, 0, 0]]
    assert padded.token_mask.tolist() == [[True, True, False, False]]


def test_pad_tokenized_batch_rejects_oversized_batches() -> None:
    with pytest.raises(ValueError, match="subword count"):
        pad_tokenized_batch(
            _tokenized_batch(),
            shapes=FixedExportShapes(
                batch_size=1,
                subword_count=3,
                token_count=4,
                character_count=None,
            ),
            padding_token_id=7,
        )
    with pytest.raises(ValueError, match="token count"):
        pad_tokenized_batch(
            _tokenized_batch(),
            shapes=FixedExportShapes(
                batch_size=1,
                subword_count=6,
                token_count=1,
                character_count=None,
            ),
            padding_token_id=7,
        )
    with pytest.raises(ValueError, match="batch size"):
        pad_tokenized_batch(
            _tokenized_batch(),
            shapes=FixedExportShapes(
                batch_size=2,
                subword_count=6,
                token_count=4,
                character_count=None,
            ),
            padding_token_id=7,
        )


def test_pad_character_token_batch_pads_token_dimension() -> None:
    padded = pad_character_token_batch(
        _character_batch(),
        shapes=_shapes(),
        character_padding_id=0,
    )

    assert padded.character_ids.shape == (1, 4, 5)
    assert padded.character_ids[0, 2].tolist() == [0, 0, 0, 0, 0]
    assert padded.character_mask[0, 2].tolist() == [False] * 5
    assert padded.token_mask.tolist() == [[True, True, False, False]]


def test_pad_character_token_batch_requires_matching_character_count() -> None:
    with pytest.raises(ValueError, match="character count"):
        pad_character_token_batch(
            _character_batch(),
            shapes=FixedExportShapes(
                batch_size=1,
                subword_count=6,
                token_count=4,
                character_count=6,
            ),
            character_padding_id=0,
        )
    with pytest.raises(ValueError, match="character count"):
        pad_character_token_batch(
            _character_batch(),
            shapes=_shapes(character_count=None),
            character_padding_id=0,
        )
