"""Filtered silver training sentences and their soft-target batches.

This module turns a prepared silver corpus plus its label artifact into
training-ready sentences. The predeclared v1 filter policy is applied here,
at load time: per task group (UPOS, complete morphology bundle, lemma top-1)
a token keeps its silver supervision only when the two teachers agree, an
optional calibrated-confidence floor can tighten that further, and sentences
with too many masked tokens are discarded completely. Because the label
artifact stores raw predictions, changing this policy never requires
relabeling.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from prism.data.examples import PretokenizedSentence
from prism.data.silver import iter_pretokenized_silver_sentences
from prism.modeling import TokenizedBatch, tokenize_pretokenized_sentences
from prism.modeling.character_batches import (
    CharacterTokenBatch,
    encode_character_token_batch,
)
from prism.schema import CharacterVocabularySchema, MorphologySchema
from prism.training.silver_labeling import (
    SilverSentenceLabels,
    iter_silver_label_records,
    load_silver_label_manifest,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverFilterPolicy:
    require_agreement: bool = True
    minimum_confidence: float = 0.0
    maximum_masked_token_ratio: float = 0.3

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_confidence) or not (
            0.0 <= self.minimum_confidence < 1.0
        ):
            raise ValueError("Minimum confidence must be in [0, 1).")
        if not math.isfinite(self.maximum_masked_token_ratio) or not (
            0.0 <= self.maximum_masked_token_ratio <= 1.0
        ):
            raise ValueError("Maximum masked-token ratio must be in [0, 1].")


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverTrainingSentence:
    """One filtered silver sentence with soft targets and task masks."""

    model_input: PretokenizedSentence
    upos_probabilities: Tensor
    morphology_probabilities: tuple[Tensor, ...]
    lemma_rule_ids: Tensor
    lemma_rule_probabilities: Tensor
    upos_mask: Tensor
    morphology_mask: Tensor
    lemma_mask: Tensor

    def __post_init__(self) -> None:
        token_count = len(self.model_input.tokens)
        if self.upos_probabilities.shape[0] != token_count:
            raise ValueError("Silver soft targets must cover every token.")
        for mask in (self.upos_mask, self.morphology_mask, self.lemma_mask):
            if mask.shape != (token_count,) or mask.dtype != torch.bool:
                raise ValueError("Silver task masks must be boolean per token.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverLoadReport:
    sentence_count: int
    retained_sentence_count: int
    retained_token_count: int
    upos_masked_ratio: float
    morphology_masked_ratio: float
    lemma_masked_ratio: float


def _decode_labeler_morphology(
    probabilities: Tensor,
    *,
    allows_multiple_values: bool,
) -> tuple[Tensor, Tensor]:
    """Return the decoded full-label-space prediction and its confidence."""

    values = probabilities.float()
    if allows_multiple_values:
        decided = values >= 0.5
        none = ~decided.any(dim=-1, keepdim=True)
        decoded = torch.cat((none, decided), dim=-1)
        confidence = torch.where(decided, values, 1.0 - values).min(dim=-1).values
    else:
        decoded = torch.zeros_like(values, dtype=torch.bool)
        decoded.scatter_(-1, values.argmax(dim=-1, keepdim=True), True)
        confidence = values.max(dim=-1).values
    return decoded, confidence


def _sentence_from_record(
    *,
    sentence: PretokenizedSentence,
    record: SilverSentenceLabels,
    morphology_schema: MorphologySchema,
    policy: SilverFilterPolicy,
) -> SilverTrainingSentence | None:
    token_count = len(sentence.tokens)
    if record.token_count != token_count:
        raise ValueError(
            "Silver labels do not align with the corpus sentence: "
            f"{record.document_id!r} #{record.sentence_index}."
        )
    if policy.require_agreement and record.agreement_upos_ids is None:
        raise ValueError(
            "Agreement filtering requires labels with agreement predictions."
        )

    upos_probabilities = record.upos_probabilities.float()
    upos_confidence, upos_prediction = upos_probabilities.max(dim=-1)
    lemma_confidence = record.lemma_rule_probabilities[:, 0].float()

    upos_mask = torch.ones(token_count, dtype=torch.bool)
    morphology_mask = torch.ones(token_count, dtype=torch.bool)
    lemma_mask = torch.ones(token_count, dtype=torch.bool)

    morphology_confidence = torch.full((token_count,), math.inf)
    for feature, probabilities in zip(
        morphology_schema.features,
        record.morphology_probabilities,
        strict=True,
    ):
        _, feature_confidence = _decode_labeler_morphology(
            probabilities,
            allows_multiple_values=feature.allows_multiple_values,
        )
        morphology_confidence = torch.minimum(
            morphology_confidence,
            feature_confidence,
        )

    if policy.require_agreement:
        assert record.agreement_upos_ids is not None
        assert record.agreement_lemma_rule_ids is not None
        assert record.agreement_morphology_predictions is not None
        upos_mask &= upos_prediction.to(torch.int16) == record.agreement_upos_ids
        lemma_mask &= record.lemma_rule_ids[:, 0] == record.agreement_lemma_rule_ids
        for feature, probabilities, agreement in zip(
            morphology_schema.features,
            record.morphology_probabilities,
            record.agreement_morphology_predictions,
            strict=True,
        ):
            decoded, _ = _decode_labeler_morphology(
                probabilities,
                allows_multiple_values=feature.allows_multiple_values,
            )
            morphology_mask &= (decoded == agreement).all(dim=-1)

    if policy.minimum_confidence > 0.0:
        upos_mask &= upos_confidence >= policy.minimum_confidence
        lemma_mask &= lemma_confidence >= policy.minimum_confidence
        morphology_mask &= morphology_confidence >= policy.minimum_confidence

    masked = ~(upos_mask & morphology_mask & lemma_mask)
    if float(masked.float().mean()) > policy.maximum_masked_token_ratio:
        return None

    return SilverTrainingSentence(
        model_input=sentence,
        upos_probabilities=record.upos_probabilities,
        morphology_probabilities=record.morphology_probabilities,
        lemma_rule_ids=record.lemma_rule_ids,
        lemma_rule_probabilities=record.lemma_rule_probabilities,
        upos_mask=upos_mask,
        morphology_mask=morphology_mask,
        lemma_mask=lemma_mask,
    )


def load_silver_training_sentences(
    *,
    corpus_path: Path,
    labels_directory: Path,
    morphology_schema: MorphologySchema,
    policy: SilverFilterPolicy,
) -> tuple[tuple[SilverTrainingSentence, ...], SilverLoadReport]:
    """Align the corpus prefix with its labels and apply the filter policy."""

    manifest = load_silver_label_manifest(labels_directory / "labels-manifest.json")
    corpus_sentences = iter_pretokenized_silver_sentences(corpus_path)

    retained: list[SilverTrainingSentence] = []
    sentence_count = 0
    retained_token_count = 0
    masked_counts = torch.zeros(3, dtype=torch.long)

    for record in iter_silver_label_records(
        directory=labels_directory,
        manifest=manifest,
    ):
        try:
            corpus_sentence = next(corpus_sentences)
        except StopIteration as error:
            raise ValueError(
                "Silver corpus ended before its label records."
            ) from error
        if (
            corpus_sentence.document_id != record.document_id
            or corpus_sentence.sentence_index != record.sentence_index
        ):
            raise ValueError(
                "Silver labels are not aligned with the corpus order at "
                f"{record.document_id!r} #{record.sentence_index}."
            )
        sentence_count += 1
        training_sentence = _sentence_from_record(
            sentence=corpus_sentence.model_input,
            record=record,
            morphology_schema=morphology_schema,
            policy=policy,
        )
        if training_sentence is None:
            continue
        retained.append(training_sentence)
        retained_token_count += record.token_count
        masked_counts[0] += int((~training_sentence.upos_mask).sum())
        masked_counts[1] += int((~training_sentence.morphology_mask).sum())
        masked_counts[2] += int((~training_sentence.lemma_mask).sum())

    if not retained:
        raise ValueError("Silver filtering retained no sentences.")

    report = SilverLoadReport(
        sentence_count=sentence_count,
        retained_sentence_count=len(retained),
        retained_token_count=retained_token_count,
        upos_masked_ratio=float(masked_counts[0]) / retained_token_count,
        morphology_masked_ratio=float(masked_counts[1]) / retained_token_count,
        lemma_masked_ratio=float(masked_counts[2]) / retained_token_count,
    )
    return tuple(retained), report


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverTokenTaskBatch:
    """One padded batch of silver sentences with soft targets and masks."""

    model_inputs: TokenizedBatch
    character_inputs: CharacterTokenBatch | None
    upos_probabilities: Tensor
    morphology_probabilities: tuple[Tensor, ...]
    lemma_rule_ids: Tensor
    lemma_rule_probabilities: Tensor
    upos_mask: Tensor
    morphology_mask: Tensor
    lemma_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.model_inputs.batch_size

    def to(self, device: torch.device) -> "SilverTokenTaskBatch":
        return SilverTokenTaskBatch(
            model_inputs=self.model_inputs.to(device),
            character_inputs=(
                None
                if self.character_inputs is None
                else self.character_inputs.to(device)
            ),
            upos_probabilities=self.upos_probabilities.to(device),
            morphology_probabilities=tuple(
                probabilities.to(device)
                for probabilities in self.morphology_probabilities
            ),
            lemma_rule_ids=self.lemma_rule_ids.to(device),
            lemma_rule_probabilities=self.lemma_rule_probabilities.to(device),
            upos_mask=self.upos_mask.to(device),
            morphology_mask=self.morphology_mask.to(device),
            lemma_mask=self.lemma_mask.to(device),
        )


def _pad_stack(rows: Sequence[Tensor], max_token_count: int) -> Tensor:
    first = rows[0]
    padded_shape = (len(rows), max_token_count, *first.shape[1:])
    padded = torch.zeros(padded_shape, dtype=first.dtype)
    for index, row in enumerate(rows):
        padded[index, : row.shape[0]] = row
    return padded


def build_silver_token_task_batch(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sentences: Sequence[SilverTrainingSentence],
    character_vocabulary: CharacterVocabularySchema | None = None,
    maximum_character_count: int = 32,
) -> SilverTokenTaskBatch:
    if not sentences:
        raise ValueError("Silver token-task batch must contain sentences.")

    model_inputs = tokenize_pretokenized_sentences(
        tokenizer=tokenizer,
        sentences=tuple(sentence.model_input for sentence in sentences),
    )
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
    max_token_count = model_inputs.max_token_count
    feature_count = len(sentences[0].morphology_probabilities)

    return SilverTokenTaskBatch(
        model_inputs=model_inputs,
        character_inputs=character_inputs,
        upos_probabilities=_pad_stack(
            tuple(sentence.upos_probabilities.float() for sentence in sentences),
            max_token_count,
        ),
        morphology_probabilities=tuple(
            _pad_stack(
                tuple(
                    sentence.morphology_probabilities[feature_index].float()
                    for sentence in sentences
                ),
                max_token_count,
            )
            for feature_index in range(feature_count)
        ),
        lemma_rule_ids=_pad_stack(
            tuple(sentence.lemma_rule_ids.long() for sentence in sentences),
            max_token_count,
        ),
        lemma_rule_probabilities=_pad_stack(
            tuple(
                sentence.lemma_rule_probabilities.float()
                for sentence in sentences
            ),
            max_token_count,
        ),
        upos_mask=_pad_stack(
            tuple(sentence.upos_mask for sentence in sentences),
            max_token_count,
        ),
        morphology_mask=_pad_stack(
            tuple(sentence.morphology_mask for sentence in sentences),
            max_token_count,
        ),
        lemma_mask=_pad_stack(
            tuple(sentence.lemma_mask for sentence in sentences),
            max_token_count,
        ),
    )


def build_silver_sentence_batches(
    *,
    sentences: Sequence[SilverTrainingSentence],
    batch_size: int,
    random_seed: int,
    epoch_index: int,
) -> tuple[tuple[SilverTrainingSentence, ...], ...]:
    if not sentences:
        raise ValueError("Silver sentence batching requires sentences.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    generator = torch.Generator()
    generator.manual_seed(random_seed + epoch_index)
    permutation = torch.randperm(len(sentences), generator=generator)
    shuffled = tuple(sentences[int(index.item())] for index in permutation)
    return tuple(
        shuffled[start : start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    )


def iter_silver_token_task_batches(
    *,
    tokenizer: PreTrainedTokenizerBase,
    sentence_batches: Sequence[Sequence[SilverTrainingSentence]],
    character_vocabulary: CharacterVocabularySchema | None = None,
    maximum_character_count: int = 32,
) -> Iterator[SilverTokenTaskBatch]:
    for sentence_batch in sentence_batches:
        yield build_silver_token_task_batch(
            tokenizer=tokenizer,
            sentences=sentence_batch,
            character_vocabulary=character_vocabulary,
            maximum_character_count=maximum_character_count,
        )
