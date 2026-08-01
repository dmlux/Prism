"""Fixed export shapes for portable runtime artifacts.

The first ExecuTorch artifact family is captured with static tensor shapes.
Native runtimes therefore pad every batch to one documented shape instead of
re-exporting per input size. This module owns that padding in one tested
place so the Python export, the parity fixtures, and future native runtimes
share the same convention:

- subword padding uses the tokenizer's padding ID with a False attention mask;
- padded token positions use the empty alignment span ``(0, 0)`` with a False
  token mask, exactly like dynamic training batches;
- padded character positions use the character padding ID with False masks;
- a batch with fewer sentences than the fixed batch size repeats its last
  sentence, because a fully padded sentence row would leave the backbone
  attention without any active position.
"""

from dataclasses import dataclass
from typing import TypeVar

import torch

from prism.modeling import CharacterTokenBatch, TokenizedBatch


SentenceT = TypeVar("SentenceT")


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedExportShapes:
    """The static tensor shapes of one exported token-tagger artifact."""

    batch_size: int
    subword_count: int
    token_count: int
    character_count: int | None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("Export batch size must be positive.")
        if self.subword_count < 3:
            raise ValueError(
                "Export subword count must fit special tokens and content."
            )
        if self.token_count < 1:
            raise ValueError("Export token count must be positive.")
        if self.character_count is not None and self.character_count < 5:
            raise ValueError(
                "Export character count must fit boundaries, token content, "
                "and truncation."
            )


def repeat_pad_sentences(
    sentences: tuple[SentenceT, ...],
    *,
    batch_size: int,
) -> tuple[SentenceT, ...]:
    """Fill a partial sentence batch to the fixed batch size."""

    if not sentences:
        raise ValueError("Sentence batch must contain sentences.")
    if len(sentences) > batch_size:
        raise ValueError(
            f"Sentence batch of size {len(sentences)} exceeds the fixed "
            f"export batch size {batch_size}."
        )

    return sentences + (sentences[-1],) * (batch_size - len(sentences))


def pad_tokenized_batch(
    batch: TokenizedBatch,
    *,
    shapes: FixedExportShapes,
    padding_token_id: int,
) -> TokenizedBatch:
    """Pad a dynamically shaped tokenized batch to the fixed export shapes."""

    if padding_token_id < 0:
        raise ValueError("Padding token ID must not be negative.")
    if batch.batch_size != shapes.batch_size:
        raise ValueError(
            f"Batch size {batch.batch_size} does not match the fixed export "
            f"batch size {shapes.batch_size}; fill partial batches with "
            "repeat_pad_sentences before tokenization."
        )
    if batch.max_subword_count > shapes.subword_count:
        raise ValueError(
            f"Batch subword count {batch.max_subword_count} exceeds the "
            f"fixed export subword count {shapes.subword_count}."
        )
    if batch.max_token_count > shapes.token_count:
        raise ValueError(
            f"Batch token count {batch.max_token_count} exceeds the fixed "
            f"export token count {shapes.token_count}."
        )

    subword_padding = shapes.subword_count - batch.max_subword_count
    token_padding = shapes.token_count - batch.max_token_count

    return TokenizedBatch(
        input_ids=torch.nn.functional.pad(
            batch.input_ids,
            (0, subword_padding),
            value=padding_token_id,
        ),
        attention_mask=torch.nn.functional.pad(
            batch.attention_mask,
            (0, subword_padding),
            value=False,
        ),
        first_subword_indices=torch.nn.functional.pad(
            batch.first_subword_indices,
            (0, token_padding),
            value=0,
        ),
        subword_end_indices=torch.nn.functional.pad(
            batch.subword_end_indices,
            (0, token_padding),
            value=0,
        ),
        token_mask=torch.nn.functional.pad(
            batch.token_mask,
            (0, token_padding),
            value=False,
        ),
    )


def pad_character_token_batch(
    batch: CharacterTokenBatch,
    *,
    shapes: FixedExportShapes,
    character_padding_id: int,
) -> CharacterTokenBatch:
    """Pad a dynamically shaped character batch to the fixed export shapes."""

    if shapes.character_count is None:
        raise ValueError(
            "Character batch padding requires export shapes with a character count."
        )
    if character_padding_id < 0:
        raise ValueError("Character padding ID must not be negative.")

    batch_size, token_count, character_count = batch.character_ids.shape
    if batch_size != shapes.batch_size:
        raise ValueError(
            f"Character batch size {batch_size} does not match the fixed "
            f"export batch size {shapes.batch_size}."
        )
    if token_count > shapes.token_count:
        raise ValueError(
            f"Character batch token count {token_count} exceeds the fixed "
            f"export token count {shapes.token_count}."
        )
    if character_count != shapes.character_count:
        raise ValueError(
            f"Character batch character count {character_count} must equal "
            f"the fixed export character count {shapes.character_count}; "
            "encoding already produces the fixed width."
        )

    token_padding = shapes.token_count - token_count

    return CharacterTokenBatch(
        character_ids=torch.nn.functional.pad(
            batch.character_ids,
            (0, 0, 0, token_padding),
            value=character_padding_id,
        ),
        character_mask=torch.nn.functional.pad(
            batch.character_mask,
            (0, 0, 0, token_padding),
            value=False,
        ),
        token_mask=torch.nn.functional.pad(
            batch.token_mask,
            (0, token_padding),
            value=False,
        ),
    )
