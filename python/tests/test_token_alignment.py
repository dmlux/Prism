import pytest
import torch

from prism.modeling import (
    build_padded_token_alignment,
    find_first_subword_indices,
)


def test_find_first_subword_indices_maps_tokens_to_subwords() -> None:
    indices = find_first_subword_indices(
        word_ids=(None, 0, 1, 1, 2, None),
        token_count=3,
    )

    assert indices == (1, 2, 4)


def test_find_first_subword_indices_rejects_missing_tokens() -> None:
    with pytest.raises(
        ValueError,
        match=r"Tokenizer output is missing token IDs: \(1,\)",
    ):
        find_first_subword_indices(
            word_ids=(None, 0, None),
            token_count=2,
        )


def test_build_padded_token_alignment_builds_token_mask() -> None:
    first_subword_indices, token_mask = build_padded_token_alignment(
        sentence_indices=(
            (1, 2, 4),
            (1,),
        )
    )

    assert torch.equal(
        first_subword_indices,
        torch.tensor(
            [
                [1, 2, 4],
                [1, 0, 0],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        token_mask,
        torch.tensor(
            [
                [True, True, True],
                [True, False, False],
            ],
            dtype=torch.bool,
        ),
    )
