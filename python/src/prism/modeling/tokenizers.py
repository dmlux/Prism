from transformers import AutoTokenizer, PreTrainedTokenizerBase
from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import Tensor

from prism.data.examples import PretokenizedSentence
from prism.modeling.alignment import (
    build_padded_token_alignment,
    find_subword_spans,
)
from prism.modeling.batches import TokenizedBatch
from prism.modeling.backbones import PretrainedBackboneSpec


def prepare_pretokenized_words(
    *,
    tokens: Sequence[str],
    has_space_before: Sequence[bool],
) -> tuple[str, ...]:
    if not tokens:
        raise ValueError("Pretokenized input must contain tokens.")
    if len(tokens) != len(has_space_before):
        raise ValueError("Token and spacing counts must match.")
    if has_space_before[0]:
        raise ValueError("The first token cannot have preceding whitespace.")

    return tuple(
        f" {token}" if space_before else token
        for token, space_before in zip(
            tokens,
            has_space_before,
            strict=True,
        )
    )


def tokenize_pretokenized_sentences(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[PretokenizedSentence],
) -> TokenizedBatch:
    if not sentences:
        raise ValueError("Tokenizer batch must contain sentences.")

    prepare_sentences = [
        list(
            prepare_pretokenized_words(
                tokens=sentence.tokens,
                has_space_before=sentence.has_space_before,
            )
        )
        for sentence in sentences
    ]

    encoded = tokenizer(
        prepare_sentences,
        is_split_into_words=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if not isinstance(input_ids, Tensor):
        raise ValueError("Tokenizer input IDs must be a PyTorch tensor.")
    if not isinstance(attention_mask, Tensor):
        raise ValueError("Tokenizer attention mask must be a PyTorch tensor.")

    sentence_spans: list[tuple[tuple[int, int], ...]] = []

    for batch_index, sentence in enumerate(sentences):
        word_ids = encoded.word_ids(batch_index=batch_index)
        if word_ids is None:
            raise ValueError(
                f"Tokenizer did not provide word IDs for batch {batch_index}."
            )

        sentence_spans.append(
            find_subword_spans(
                word_ids=word_ids,
                token_count=len(sentence.tokens),
            )
        )

    first_subword_indices, subword_end_indices, token_mask = (
        build_padded_token_alignment(
            sentence_spans=sentence_spans,
        )
    )

    return TokenizedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask.to(dtype=torch.bool),
        first_subword_indices=first_subword_indices,
        subword_end_indices=subword_end_indices,
        token_mask=token_mask,
    )


def load_backbone_tokenizer(
    spec: PretrainedBackboneSpec,
) -> PreTrainedTokenizerBase:
    from_pretrained = cast(
        Callable[..., PreTrainedTokenizerBase | None],
        AutoTokenizer.from_pretrained,
    )
    tokenizer = from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )

    if tokenizer is None:
        raise RuntimeError("Backbone tokenizer could not be loaded.")

    if not tokenizer.is_fast:
        raise ValueError("Backbone tokenizer must be a fast tokenizer.")

    return tokenizer
