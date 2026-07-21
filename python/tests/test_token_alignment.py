import pytest
import torch

from prism.modeling import (
    ContextualizedSubwordBatch,
    TokenizedBatch,
    TokenPoolingStrategy,
    align_subwords_to_tokens,
    build_padded_token_alignment,
    find_subword_spans,
)


def test_find_subword_spans_maps_tokens_to_complete_subword_ranges() -> None:
    spans = find_subword_spans(
        word_ids=(None, 0, 1, 1, 2, None),
        token_count=3,
    )

    assert spans == ((1, 2), (2, 4), (4, 5))


def test_find_subword_spans_rejects_missing_tokens() -> None:
    with pytest.raises(
        ValueError,
        match=r"Tokenizer output is missing token IDs: \(1,\)",
    ):
        find_subword_spans(
            word_ids=(None, 0, None),
            token_count=2,
        )


def test_find_subword_spans_rejects_non_contiguous_subwords() -> None:
    with pytest.raises(
        ValueError,
        match="Subwords for token ID 0 must be contiguous",
    ):
        find_subword_spans(
            word_ids=(None, 0, 1, 0, None),
            token_count=2,
        )


def test_build_padded_token_alignment_builds_token_mask() -> None:
    first_subword_indices, subword_end_indices, token_mask = (
        build_padded_token_alignment(
            sentence_spans=(
                ((1, 2), (2, 4), (4, 5)),
                ((1, 2),),
            )
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
        subword_end_indices,
        torch.tensor(
            [
                [2, 4, 5],
                [2, 0, 0],
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
        subword_end_indices=torch.tensor(
            [
                [2, 4],
                [3, 0],
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


def test_align_subwords_to_tokens_means_complete_subword_spans() -> None:
    tokenized_batch = TokenizedBatch(
        input_ids=torch.tensor([[1, 10, 20, 21, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[True, True, True, True, True]]),
        first_subword_indices=torch.tensor([[1, 2]], dtype=torch.long),
        subword_end_indices=torch.tensor([[2, 4]], dtype=torch.long),
        token_mask=torch.tensor([[True, True]]),
    )
    subword_batch = ContextualizedSubwordBatch(
        hidden_states=torch.tensor(
            [[[0.0], [2.0], [4.0], [8.0], [10.0]]],
            dtype=torch.float32,
        )
    )

    token_batch = align_subwords_to_tokens(
        subword_batch=subword_batch,
        tokenized_batch=tokenized_batch,
        pooling_strategy=TokenPoolingStrategy.MEAN,
    )

    assert torch.equal(
        token_batch.hidden_states,
        torch.tensor([[[2.0], [6.0]]], dtype=torch.float32),
    )
