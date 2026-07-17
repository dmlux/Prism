import torch
from torch import Tensor

from collections.abc import Sequence


def find_first_subword_indices(
    *,
    word_ids: Sequence[int | None],
    token_count: int,
) -> tuple[int, ...]:
    if token_count <= 0:
        raise ValueError("Token count must be positive.")

    first_indices = [-1] * token_count

    for subword_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id < 0 or word_id >= token_count:
            raise ValueError(f"Word ID is outside the token range: {word_id}")
        if first_indices[word_id] == -1:
            first_indices[word_id] = subword_index

    missing_token_ids = tuple(
        token_id
        for token_id, subword_index in enumerate(first_indices)
        if subword_index == -1
    )
    if missing_token_ids:
        raise ValueError(f"Tokenizer output is missing token IDs: {missing_token_ids}")

    return tuple(first_indices)


def build_padded_token_alignment(
    *,
    sentence_indices: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    if not sentence_indices:
        raise ValueError("Token alignment batch must contain sentences.")
    if any(not indices for indices in sentence_indices):
        raise ValueError("Every sentence must contain token indices.")
    if any(index < 0 for indices in sentence_indices for index in indices):
        raise ValueError("Subword indices must not be negative.")

    max_token_count = max(len(indices) for indices in sentence_indices)

    padded_indices = tuple(
        tuple(indices) + (0,) * (max_token_count - len(indices))
        for indices in sentence_indices
    )
    token_mask = tuple(
        (True,) * len(indices) + (False,) * (max_token_count - len(indices))
        for indices in sentence_indices
    )

    return (
        torch.tensor(padded_indices, dtype=torch.long),
        torch.tensor(token_mask, dtype=torch.bool),
    )
