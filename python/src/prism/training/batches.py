from dataclasses import dataclass
from collections.abc import Iterable, Iterator, Sequence

from transformers import PreTrainedTokenizerBase

import torch

from prism.data import (
    SupervisedSentence,
    TokenTaskTargetBatch,
    build_token_task_target_batch,
)
from prism.modeling import TokenizedBatch, tokenize_pretokenized_sentences


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedTokenTaskBatch:
    model_inputs: TokenizedBatch
    targets: TokenTaskTargetBatch

    def __post_init__(self) -> None:
        if self.model_inputs.batch_size != self.targets.batch_size:
            raise ValueError("Model inputs and targets must have the same batch size.")
        if self.model_inputs.max_token_count != self.targets.max_token_count:
            raise ValueError("Model inputs and targets must have the same token count.")
        if not torch.equal(
            self.model_inputs.token_mask,
            self.targets.token_mask,
        ):
            raise ValueError(
                "Model inputs and targets must have identical token masks."
            )

    @property
    def batch_size(self) -> int:
        return self.model_inputs.batch_size

    @property
    def max_token_count(self) -> int:
        return self.model_inputs.max_token_count

    def to(self, device: torch.device) -> "SupervisedTokenTaskBatch":
        return SupervisedTokenTaskBatch(
            model_inputs=self.model_inputs.to(device),
            targets=self.targets.to(device),
        )


def build_supervised_token_task_batch(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[SupervisedSentence],
) -> SupervisedTokenTaskBatch:
    if not sentences:
        raise ValueError("Supervised token-task batch must contain sentences.")

    model_inputs = tokenize_pretokenized_sentences(
        tokenizer=tokenizer,
        sentences=tuple(sentence.model_input for sentence in sentences),
    )
    targets = build_token_task_target_batch(sentences)

    return SupervisedTokenTaskBatch(
        model_inputs=model_inputs,
        targets=targets,
    )


def build_supervised_sentence_batches(
    *,
    sentences: Sequence[SupervisedSentence],
    batch_size: int,
    random_seed: int,
    epoch_index: int,
) -> tuple[tuple[SupervisedSentence, ...], ...]:
    if not sentences:
        raise ValueError("Sentence batching requires supervised sentences.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    if random_seed < 0:
        raise ValueError("Random seed must be non-negative.")
    if epoch_index < 0:
        raise ValueError("Epoch index must be non-negative.")

    generator = torch.Generator()
    generator.manual_seed(random_seed + epoch_index)

    permutation = torch.randperm(
        len(sentences),
        generator=generator,
    )
    shuffled_indices = tuple(int(index.item()) for index in permutation)
    shuffled_sentences = tuple(sentences[index] for index in shuffled_indices)

    return tuple(
        shuffled_sentences[start : start + batch_size]
        for start in range(0, len(shuffled_sentences), batch_size)
    )


def iter_supervised_token_task_batches(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sentence_batches: Iterable[Sequence[SupervisedSentence]],
) -> Iterator[SupervisedTokenTaskBatch]:
    for sentence_batch in sentence_batches:
        yield build_supervised_token_task_batch(
            tokenizer=tokenizer,
            sentences=sentence_batch,
        )
