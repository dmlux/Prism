from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from prism.schema.characters import (
    CHARACTER_END_ID,
    CHARACTER_FIRST_LITERAL_ID,
    CHARACTER_PADDING_ID,
    CHARACTER_START_ID,
    CHARACTER_TRUNCATION_ID,
    CHARACTER_UNKNOWN_ID,
    CharacterVocabularySchema,
    normalize_character_token,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CharacterTokenBatch:
    character_ids: Tensor
    character_mask: Tensor
    token_mask: Tensor

    def __post_init__(self) -> None:
        if self.character_ids.ndim != 3:
            raise ValueError("Character IDs must have three dimensions.")
        if self.character_ids.dtype != torch.long:
            raise ValueError("Character IDs must use torch.long.")
        if self.character_mask.shape != self.character_ids.shape:
            raise ValueError("Character mask must match character IDs.")
        if self.character_mask.dtype != torch.bool:
            raise ValueError("Character mask must use torch.bool.")
        if self.token_mask.shape != self.character_ids.shape[:2]:
            raise ValueError("Token mask must match character token dimensions.")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("Token mask must use torch.bool.")
        if not (
            self.character_ids.device
            == self.character_mask.device
            == self.token_mask.device
        ):
            raise ValueError("Character batch tensors must use the same device.")

    def to(self, device: torch.device) -> "CharacterTokenBatch":
        return CharacterTokenBatch(
            character_ids=self.character_ids.to(device=device),
            character_mask=self.character_mask.to(device=device),
            token_mask=self.token_mask.to(device=device),
        )


def encode_character_token_batch(
    *,
    token_sequences: Sequence[Sequence[str]],
    vocabulary: CharacterVocabularySchema,
    maximum_character_count: int,
) -> CharacterTokenBatch:
    if not token_sequences or any(not tokens for tokens in token_sequences):
        raise ValueError("Character batch must contain non-empty token sequences.")
    if maximum_character_count < 5:
        raise ValueError(
            "Maximum character count must fit boundaries, token content, and "
            "truncation."
        )

    normalized_sequences = tuple(
        tuple(normalize_character_token(token) for token in tokens)
        for tokens in token_sequences
    )
    character_ids_by_value = {
        character: character_id
        for character_id, character in enumerate(
            vocabulary.characters,
            start=CHARACTER_FIRST_LITERAL_ID,
        )
    }
    maximum_token_count = max(len(tokens) for tokens in normalized_sequences)
    character_ids = torch.full(
        (
            len(normalized_sequences),
            maximum_token_count,
            maximum_character_count,
        ),
        fill_value=CHARACTER_PADDING_ID,
        dtype=torch.long,
    )
    character_mask = torch.zeros_like(character_ids, dtype=torch.bool)
    token_mask = torch.zeros(
        (len(normalized_sequences), maximum_token_count),
        dtype=torch.bool,
    )

    for sentence_index, tokens in enumerate(normalized_sequences):
        for token_index, token in enumerate(tokens):
            encoded = _encode_token(
                token=token,
                character_ids_by_value=character_ids_by_value,
                maximum_character_count=maximum_character_count,
            )
            encoded_length = len(encoded)
            character_ids[
                sentence_index,
                token_index,
                :encoded_length,
            ] = torch.tensor(encoded, dtype=torch.long)
            character_mask[
                sentence_index,
                token_index,
                :encoded_length,
            ] = True
            token_mask[sentence_index, token_index] = True

    return CharacterTokenBatch(
        character_ids=character_ids,
        character_mask=character_mask,
        token_mask=token_mask,
    )


def _encode_token(
    *,
    token: str,
    character_ids_by_value: dict[str, int],
    maximum_character_count: int,
) -> tuple[int, ...]:
    literal_ids = tuple(
        character_ids_by_value.get(character, CHARACTER_UNKNOWN_ID)
        for character in token
    )
    complete_ids = (CHARACTER_START_ID, *literal_ids, CHARACTER_END_ID)
    if len(complete_ids) <= maximum_character_count:
        return complete_ids

    retained_literal_count = maximum_character_count - 3
    prefix_count = (retained_literal_count + 1) // 2
    suffix_count = retained_literal_count - prefix_count

    return (
        CHARACTER_START_ID,
        *literal_ids[:prefix_count],
        CHARACTER_TRUNCATION_ID,
        *literal_ids[-suffix_count:],
        CHARACTER_END_ID,
    )
