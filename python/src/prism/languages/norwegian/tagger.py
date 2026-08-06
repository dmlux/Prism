"""Offline runtime tagging API for host applications.

One tagger, two entry points: ``tag_text`` runs the recall-oriented runtime
segmentation first (nothing the user wrote is ever dropped; over-long
sentences are chunked), ``tag_pretokenized`` accepts tokens the application
already has. Both return the same result: tokens with UPOS, morphology
features, lemma, and **calibrated** confidences per decision, using the
frozen production decoding policy (morphology-logit correction plus per-head
temperatures from the checkpoint's calibration artifact).
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from prism.data import PretokenizedSentence
from prism.data.segmentation import segment_pretokenized_sentences
from prism.languages.norwegian.checkpoint_loading import (
    load_norwegian_token_tagger,
)
from prism.languages.norwegian.silver_extraction import (
    norwegian_sentence_extraction_policy,
)
from prism.modeling import tokenize_pretokenized_sentences
from prism.modeling.character_batches import encode_character_token_batch
from prism.modeling.decoding import apply_morphology_logit_correction
from prism.schema.morphology import NO_MORPHOLOGY_VALUE
from prism.training import morphology_logit_correction_from_checkpoint
from prism.training.calibration import (
    calibrated_task_probabilities,
    load_task_temperature_calibration,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class UposProbability:
    """One entry of a token's UPOS probability distribution."""

    upos: str
    probability: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TaggedToken:
    text: str
    has_space_before: bool
    upos: str
    upos_confidence: float
    features: Mapping[str, tuple[str, ...]]
    feature_confidences: Mapping[str, float]
    lemma: str
    lemma_confidence: float

    # The complete calibrated UPOS probability distribution: one entry per
    # label of the loaded schema, sorted by descending probability (the
    # first entry is the decision reported by upos and upos_confidence).
    upos_distribution: tuple[UposProbability, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class TaggedSentence:
    tokens: tuple[TaggedToken, ...]


class NorwegianTagger:
    """Frozen-checkpoint Norwegian tagger with calibrated confidences."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        calibration_path: Path,
        language_tag: str = "nb",
        treebank_release: str = "current",
        device: str = "cpu",
        morphology_logit_correction_strength: float = 0.25,
        batch_size: int = 32,
        maximum_token_count: int = 128,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Tagger batch size must be positive.")
        loaded = load_norwegian_token_tagger(
            checkpoint_path=checkpoint_path,
            required_language_tags=(language_tag,),
            treebank_release=treebank_release,
        )
        calibration = load_task_temperature_calibration(calibration_path)
        if (
            calibration.morphology_logit_correction_strength
            != morphology_logit_correction_strength
        ):
            raise ValueError(
                "Calibration was fitted for a different logit-correction "
                f"strength: {calibration.morphology_logit_correction_strength}."
            )
        self._device = torch.device(device)
        self._model = loaded.model
        self._model.to(self._device)
        self._model.eval()
        self._schema = loaded.schema
        self._tokenizer = loaded.tokenizer
        self._character_vocabulary = loaded.character_vocabulary
        self._maximum_character_count = loaded.maximum_character_count
        self._calibration = calibration
        self._correction = morphology_logit_correction_from_checkpoint(
            loaded.checkpoint,
            strength=morphology_logit_correction_strength,
        )
        self._batch_size = batch_size
        self._segmentation_policy = norwegian_sentence_extraction_policy(
            maximum_token_count=maximum_token_count,
        )

    @property
    def upos_labels(self) -> tuple[str, ...]:
        """Every UPOS tag the loaded schema can assign."""

        return self._schema.upos.labels

    @property
    def morphology_features(self) -> Mapping[str, tuple[str, ...]]:
        """Every morphology feature with its possible values, in schema order."""

        return {
            feature.name: feature.values
            for feature in self._schema.morphology.features
        }

    def tag_text(self, text: str) -> tuple[TaggedSentence, ...]:
        """Segment raw text without dropping content, then tag it."""

        sentences = tuple(
            segment_pretokenized_sentences(text, self._segmentation_policy)
        )
        return self._tag_sentences(sentences)

    def tag_pretokenized(
        self,
        sentences: Sequence[Sequence[str]],
    ) -> tuple[TaggedSentence, ...]:
        """Tag sentences the application already tokenized itself."""

        pretokenized = tuple(
            PretokenizedSentence(
                tokens=tuple(tokens),
                has_space_before=(False,) + (True,) * (len(tokens) - 1),
            )
            for tokens in sentences
            if tokens
        )
        return self._tag_sentences(pretokenized)

    def _tag_sentences(
        self,
        sentences: tuple[PretokenizedSentence, ...],
    ) -> tuple[TaggedSentence, ...]:
        tagged: list[TaggedSentence] = []
        for start in range(0, len(sentences), self._batch_size):
            tagged.extend(self._tag_batch(sentences[start : start + self._batch_size]))
        return tuple(tagged)

    def _tag_batch(
        self,
        sentences: tuple[PretokenizedSentence, ...],
    ) -> Iterator[TaggedSentence]:
        model_inputs = tokenize_pretokenized_sentences(
            tokenizer=self._tokenizer,
            sentences=sentences,
        ).to(self._device)
        character_inputs = (
            None
            if self._character_vocabulary is None
            else encode_character_token_batch(
                token_sequences=tuple(sentence.tokens for sentence in sentences),
                vocabulary=self._character_vocabulary,
                maximum_character_count=self._maximum_character_count,
            ).to(self._device)
        )

        with torch.inference_mode():
            if character_inputs is None:
                logits = self._model(model_inputs)
            else:
                logits = self._model(model_inputs, character_inputs)
            if self._correction is not None:
                logits = apply_morphology_logit_correction(
                    logits=logits,
                    morphology_schema=self._schema.morphology,
                    correction=self._correction,
                )
            probabilities = calibrated_task_probabilities(
                logits=logits,
                morphology_schema=self._schema.morphology,
                calibration=self._calibration,
            )

        upos_probabilities, upos_ids = probabilities.upos_probabilities.max(dim=-1)
        lemma_probabilities, lemma_rule_ids = (
            probabilities.lemma_rule_probabilities.max(dim=-1)
        )
        for sentence_index, sentence in enumerate(sentences):
            tokens: list[TaggedToken] = []
            for token_index, token_text in enumerate(sentence.tokens):
                features: dict[str, tuple[str, ...]] = {}
                feature_confidences: dict[str, float] = {}
                for feature, feature_probabilities in zip(
                    self._schema.morphology.features,
                    probabilities.morphology_probabilities,
                    strict=True,
                ):
                    token_probabilities = feature_probabilities[
                        sentence_index, token_index
                    ]
                    if feature.allows_multiple_values:
                        selected = tuple(
                            value
                            for value, probability in zip(
                                feature.values,
                                token_probabilities.tolist(),
                                strict=True,
                            )
                            if probability > 0.5
                        )
                        if selected:
                            confidence = min(
                                probability
                                for probability in token_probabilities.tolist()
                                if probability > 0.5
                            )
                        else:
                            confidence = float(
                                (1.0 - token_probabilities).prod().item()
                            )
                    else:
                        value_index = int(token_probabilities.argmax().item())
                        confidence = float(token_probabilities[value_index].item())
                        labels = (NO_MORPHOLOGY_VALUE, *feature.values)
                        label = labels[value_index]
                        selected = () if label == NO_MORPHOLOGY_VALUE else (label,)
                    if selected:
                        features[feature.name] = selected
                        feature_confidences[feature.name] = float(confidence)

                lemma_rule = self._schema.lemma_rules.rules[
                    int(lemma_rule_ids[sentence_index, token_index].item())
                ]
                token_upos_probabilities = probabilities.upos_probabilities[
                    sentence_index, token_index
                ]
                upos_distribution = tuple(
                    sorted(
                        (
                            UposProbability(
                                upos=label,
                                probability=float(
                                    token_upos_probabilities[label_index].item()
                                ),
                            )
                            for label_index, label in enumerate(
                                self._schema.upos.labels
                            )
                        ),
                        key=lambda entry: entry.probability,
                        reverse=True,
                    )
                )
                tokens.append(
                    TaggedToken(
                        text=token_text,
                        has_space_before=sentence.has_space_before[token_index],
                        upos=self._schema.upos.labels[
                            int(upos_ids[sentence_index, token_index].item())
                        ],
                        upos_confidence=float(
                            upos_probabilities[sentence_index, token_index].item()
                        ),
                        features=features,
                        feature_confidences=feature_confidences,
                        lemma=lemma_rule.apply(token_text),
                        lemma_confidence=float(
                            lemma_probabilities[sentence_index, token_index].item()
                        ),
                        upos_distribution=upos_distribution,
                    )
                )
            yield TaggedSentence(tokens=tuple(tokens))
