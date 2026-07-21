from collections.abc import Sequence
from enum import StrEnum

import torch
from torch import Tensor

from prism.modeling.batches import TokenizedBatch
from prism.modeling.outputs import (
    ContextualizedTokenBatch,
    ContextualizedSubwordBatch,
)


class TokenPoolingStrategy(StrEnum):
    FIRST = "first"
    MEAN = "mean"


def find_subword_spans(
    *,
    word_ids: Sequence[int | None],
    token_count: int,
) -> tuple[tuple[int, int], ...]:
    if token_count <= 0:
        raise ValueError("Token count must be positive.")

    first_indices = [-1] * token_count
    end_indices = [-1] * token_count

    for subword_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id < 0 or word_id >= token_count:
            raise ValueError(f"Word ID is outside the token range: {word_id}")
        if first_indices[word_id] == -1:
            first_indices[word_id] = subword_index
        end_indices[word_id] = subword_index + 1

    missing_token_ids = tuple(
        token_id
        for token_id, subword_index in enumerate(first_indices)
        if subword_index == -1
    )
    if missing_token_ids:
        raise ValueError(f"Tokenizer output is missing token IDs: {missing_token_ids}")

    for token_id, (start, end) in enumerate(
        zip(first_indices, end_indices, strict=True)
    ):
        if any(word_id != token_id for word_id in word_ids[start:end]):
            raise ValueError(f"Subwords for token ID {token_id} must be contiguous.")

    return tuple(zip(first_indices, end_indices, strict=True))


def build_padded_token_alignment(
    *,
    sentence_spans: Sequence[Sequence[tuple[int, int]]],
) -> tuple[Tensor, Tensor, Tensor]:
    if not sentence_spans:
        raise ValueError("Token alignment batch must contain sentences.")
    if any(not spans for spans in sentence_spans):
        raise ValueError("Every sentence must contain token spans.")
    if any(
        start < 0 or end <= start for spans in sentence_spans for start, end in spans
    ):
        raise ValueError("Every subword span must be non-empty and non-negative.")

    max_token_count = max(len(spans) for spans in sentence_spans)

    padded_starts = tuple(
        tuple(start for start, _ in spans) + (0,) * (max_token_count - len(spans))
        for spans in sentence_spans
    )
    padded_ends = tuple(
        tuple(end for _, end in spans) + (0,) * (max_token_count - len(spans))
        for spans in sentence_spans
    )
    token_mask = tuple(
        (True,) * len(spans) + (False,) * (max_token_count - len(spans))
        for spans in sentence_spans
    )

    return (
        torch.tensor(padded_starts, dtype=torch.long),
        torch.tensor(padded_ends, dtype=torch.long),
        torch.tensor(token_mask, dtype=torch.bool),
    )


def align_subwords_to_tokens(
    *,
    subword_batch: ContextualizedSubwordBatch,
    tokenized_batch: TokenizedBatch,
    pooling_strategy: TokenPoolingStrategy = TokenPoolingStrategy.FIRST,
) -> ContextualizedTokenBatch:
    if subword_batch.batch_size != tokenized_batch.batch_size:
        raise ValueError("Subword and tokenized batch sizes must match.")
    if subword_batch.max_subword_count != tokenized_batch.max_subword_count:
        raise ValueError("Subword counts must match the tokenized batch.")

    if pooling_strategy is TokenPoolingStrategy.FIRST:
        gather_indices = tokenized_batch.first_subword_indices.unsqueeze(-1).expand(
            -1,
            -1,
            subword_batch.hidden_size,
        )
        token_hidden_states = torch.gather(
            subword_batch.hidden_states,
            dim=1,
            index=gather_indices,
        )
    elif pooling_strategy is TokenPoolingStrategy.MEAN:
        prefix_sums = torch.cat(
            (
                torch.zeros_like(subword_batch.hidden_states[:, :1]),
                subword_batch.hidden_states.cumsum(dim=1),
            ),
            dim=1,
        )
        start_indices = tokenized_batch.first_subword_indices.unsqueeze(-1).expand(
            -1,
            -1,
            subword_batch.hidden_size,
        )
        end_indices = tokenized_batch.subword_end_indices.unsqueeze(-1).expand(
            -1,
            -1,
            subword_batch.hidden_size,
        )
        span_sums = torch.gather(prefix_sums, dim=1, index=end_indices) - torch.gather(
            prefix_sums,
            dim=1,
            index=start_indices,
        )
        span_lengths = (
            tokenized_batch.subword_end_indices - tokenized_batch.first_subword_indices
        ).clamp_min(1)
        token_hidden_states = span_sums / span_lengths.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported token pooling strategy: {pooling_strategy!r}")
    token_hidden_states = token_hidden_states.masked_fill(
        ~tokenized_batch.token_mask.unsqueeze(-1), 0.0
    )

    return ContextualizedTokenBatch(
        hidden_states=token_hidden_states,
        token_mask=tokenized_batch.token_mask,
    )
