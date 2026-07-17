import pytest
import torch

from prism.modeling import (
    ContextualizedSubwordBatch,
    TokenizedBatch,
    align_subwords_to_tokens,
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


def test_align_subwords_to_tokens_gathers_first_subword_vectors() -> None:
    tokenized_batch = TokenizedBatch(
        input_ids=torch.tensor(
            [
                [1, 10, 20, 30],
                [1, 40, 50, 0],
            ],
            dtype=torch.long,
        ),
        attention_mask=torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
            ],
            dtype=torch.bool,
        ),
        first_subword_indices=torch.tensor(
            [
                [1, 3],
                [2, 0],
            ],
            dtype=torch.long,
        ),
        token_mask=torch.tensor(
            [
                [True, True],
                [True, False],
            ],
            dtype=torch.bool,
        ),
    )
    subword_batch = ContextualizedSubwordBatch(
        hidden_states=torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0]],
                [[10.0], [11.0], [12.0], [13.0]],
            ],
            dtype=torch.float32,
        )
    )

    token_batch = align_subwords_to_tokens(
        subword_batch=subword_batch,
        tokenized_batch=tokenized_batch,
    )

    assert torch.equal(
        token_batch.hidden_states,
        torch.tensor(
            [
                [[1.0], [3.0]],
                [[12.0], [0.0]],
            ],
            dtype=torch.float32,
        ),
    )
    assert token_batch.token_mask is tokenized_batch.token_mask
