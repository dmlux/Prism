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
from prism.modeling.character_batches import (
    CharacterTokenBatch,
    encode_character_token_batch,
)
from prism.schema import CharacterVocabularySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedTokenTaskBatch:
    model_inputs: TokenizedBatch
    targets: TokenTaskTargetBatch
    character_inputs: CharacterTokenBatch | None = None

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
        if self.character_inputs is not None and not torch.equal(
            self.character_inputs.token_mask,
            self.targets.token_mask,
        ):
            raise ValueError(
                "Character inputs and targets must have identical token masks."
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
            character_inputs=(
                None
                if self.character_inputs is None
                else self.character_inputs.to(device)
            ),
        )


def build_supervised_token_task_batch(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[SupervisedSentence],
    character_vocabulary: CharacterVocabularySchema | None = None,
    maximum_character_count: int = 32,
) -> SupervisedTokenTaskBatch:
    if not sentences:
        raise ValueError("Supervised token-task batch must contain sentences.")

    model_inputs = tokenize_pretokenized_sentences(
        tokenizer=tokenizer,
        sentences=tuple(sentence.model_input for sentence in sentences),
    )
    targets = build_token_task_target_batch(sentences)
    character_inputs = (
        None
        if character_vocabulary is None
        else encode_character_token_batch(
            token_sequences=tuple(
                sentence.model_input.tokens for sentence in sentences
            ),
            vocabulary=character_vocabulary,
            maximum_character_count=maximum_character_count,
        )
    )

    return SupervisedTokenTaskBatch(
        model_inputs=model_inputs,
        targets=targets,
        character_inputs=character_inputs,
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
    character_vocabulary: CharacterVocabularySchema | None = None,
    maximum_character_count: int = 32,
) -> Iterator[SupervisedTokenTaskBatch]:
    for sentence_batch in sentence_batches:
        yield build_supervised_token_task_batch(
            tokenizer=tokenizer,
            sentences=sentence_batch,
            character_vocabulary=character_vocabulary,
            maximum_character_count=maximum_character_count,
        )
