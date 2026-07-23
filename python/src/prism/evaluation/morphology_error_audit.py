"""Token-aligned morphology error attribution for fixed evaluation splits."""

from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

from prism.evaluation.token_frequency import (
    TokenFrequencyProfile,
    normalize_token_form,
)
from prism.evaluation.universal_dependencies import (
    UniversalDependenciesReferenceBatch,
)
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.schema import TokenTaskSchema, decode_morphology_values


CountKey = TypeVar("CountKey", bound=Hashable)


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedCount:
    name: str
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureConfusionCount:
    gold_values: tuple[str, ...]
    predicted_values: tuple[str, ...]
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyContextCount:
    previous_upos: str
    gold_upos: str
    next_upos: str
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyErrorRecord:
    sentence_index: int
    token_index: int
    form: str
    normalized_form: str
    lemma: str
    gold_upos: str
    predicted_upos: str
    gold_values: tuple[str, ...]
    predicted_values: tuple[str, ...]
    training_frequency: int
    frequency_class: str
    previous_gold_upos: str
    next_gold_upos: str
    comparison_values: tuple[str, ...] | None
    comparison_feature_correct: bool | None
    comparison_bundle_correct: bool | None


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyErrorAudit:
    feature_name: str
    token_count: int
    error_count: int
    comparison_feature_correct_count: int | None
    comparison_bundle_correct_count: int | None
    gold_upos_counts: tuple[NamedCount, ...]
    predicted_upos_counts: tuple[NamedCount, ...]
    frequency_class_counts: tuple[NamedCount, ...]
    normalized_form_counts: tuple[NamedCount, ...]
    confusion_counts: tuple[MorphologyFeatureConfusionCount, ...]
    context_counts: tuple[MorphologyContextCount, ...]
    errors: tuple[MorphologyErrorRecord, ...]


@dataclass(slots=True, kw_only=True)
class MorphologyErrorAuditAccumulator:
    schema: TokenTaskSchema
    feature_name: str
    reference_batches: tuple[UniversalDependenciesReferenceBatch, ...]
    frequency_profile: TokenFrequencyProfile
    comparison_reference_batches: (
        tuple[UniversalDependenciesReferenceBatch, ...] | None
    ) = None
    _feature_index: int = field(init=False)
    _batch_index: int = 0
    _sentence_index: int = 0
    _token_count: int = 0
    _errors: list[MorphologyErrorRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        feature_names = tuple(
            feature.name for feature in self.schema.morphology.features
        )
        try:
            self._feature_index = feature_names.index(self.feature_name)
        except ValueError as error:
            raise ValueError(
                f"Unknown morphology audit feature: {self.feature_name!r}"
            ) from error

        if not self.reference_batches:
            raise ValueError("Morphology error audit requires reference batches.")
        if self.comparison_reference_batches is not None and len(
            self.comparison_reference_batches
        ) != len(self.reference_batches):
            raise ValueError(
                "Morphology audit comparison and gold batch counts must match."
            )

    def add(self, *, predictions: TokenTaskPredictionBatch) -> None:
        if self._batch_index >= len(self.reference_batches):
            raise ValueError(
                "Morphology error audit received more predictions than references."
            )

        reference_batch = self.reference_batches[self._batch_index]
        comparison_batch = (
            None
            if self.comparison_reference_batches is None
            else self.comparison_reference_batches[self._batch_index]
        )
        if predictions.upos_ids.shape[0] != len(reference_batch.sentences):
            raise ValueError(
                "Morphology audit prediction and reference batch sizes must match."
            )
        if comparison_batch is not None and len(comparison_batch.sentences) != len(
            reference_batch.sentences
        ):
            raise ValueError(
                "Morphology audit comparison and gold sentence counts must match."
            )

        for batch_sentence_index, reference_sentence in enumerate(
            reference_batch.sentences
        ):
            predicted_token_count = int(
                predictions.token_mask[batch_sentence_index].sum().item()
            )
            if predicted_token_count != len(reference_sentence):
                raise ValueError(
                    "Morphology audit prediction and reference token counts must match."
                )
            comparison_sentence = (
                None
                if comparison_batch is None
                else comparison_batch.sentences[batch_sentence_index]
            )
            if comparison_sentence is not None and len(comparison_sentence) != len(
                reference_sentence
            ):
                raise ValueError(
                    "Morphology audit comparison and gold token counts must match."
                )

            for token_index, reference in enumerate(reference_sentence):
                comparison = (
                    None
                    if comparison_sentence is None
                    else comparison_sentence[token_index]
                )
                if comparison is not None and comparison.form != reference.form:
                    raise ValueError(
                        "Morphology audit comparison requires identical token forms."
                    )

                predicted_features = decode_morphology_values(
                    self.schema.morphology,
                    tuple(
                        tuple(
                            bool(value)
                            for value in feature_predictions[
                                batch_sentence_index, token_index
                            ].tolist()
                        )
                        for feature_predictions in predictions.morphology_predictions
                    ),
                )
                gold_values = _feature_values(
                    reference.universal_features,
                    feature_name=self.feature_name,
                )
                predicted_values = _mapping_feature_values(
                    predicted_features.get(self.feature_name)
                )
                self._token_count += 1

                if gold_values == predicted_values:
                    continue

                predicted_upos = self.schema.upos.label_for_id(
                    int(predictions.upos_ids[batch_sentence_index, token_index].item())
                )
                comparison_values = (
                    None
                    if comparison is None
                    else _feature_values(
                        comparison.universal_features,
                        feature_name=self.feature_name,
                    )
                )
                self._errors.append(
                    MorphologyErrorRecord(
                        sentence_index=self._sentence_index,
                        token_index=token_index,
                        form=reference.form,
                        normalized_form=normalize_token_form(reference.form),
                        lemma=reference.lemma,
                        gold_upos=reference.upos,
                        predicted_upos=predicted_upos,
                        gold_values=gold_values,
                        predicted_values=predicted_values,
                        training_frequency=self.frequency_profile.frequency(
                            reference.form
                        ),
                        frequency_class=self.frequency_profile.classify(
                            reference.form
                        ).value,
                        previous_gold_upos=(
                            "<BOS>"
                            if token_index == 0
                            else reference_sentence[token_index - 1].upos
                        ),
                        next_gold_upos=(
                            "<EOS>"
                            if token_index + 1 == len(reference_sentence)
                            else reference_sentence[token_index + 1].upos
                        ),
                        comparison_values=comparison_values,
                        comparison_feature_correct=(
                            None
                            if comparison is None
                            else comparison_values == gold_values
                        ),
                        comparison_bundle_correct=(
                            None
                            if comparison is None
                            else comparison.universal_features
                            == reference.universal_features
                        ),
                    )
                )

            self._sentence_index += 1

        self._batch_index += 1

    def finish(self) -> MorphologyErrorAudit:
        if self._batch_index != len(self.reference_batches):
            raise ValueError("Morphology error audit contains unused references.")
        if not self._token_count:
            raise ValueError("Morphology error audit must contain tokens.")

        errors = tuple(self._errors)
        comparison_enabled = self.comparison_reference_batches is not None
        return MorphologyErrorAudit(
            feature_name=self.feature_name,
            token_count=self._token_count,
            error_count=len(errors),
            comparison_feature_correct_count=(
                None
                if not comparison_enabled
                else sum(error.comparison_feature_correct is True for error in errors)
            ),
            comparison_bundle_correct_count=(
                None
                if not comparison_enabled
                else sum(error.comparison_bundle_correct is True for error in errors)
            ),
            gold_upos_counts=_named_counts(error.gold_upos for error in errors),
            predicted_upos_counts=_named_counts(
                error.predicted_upos for error in errors
            ),
            frequency_class_counts=_named_counts(
                error.frequency_class for error in errors
            ),
            normalized_form_counts=_named_counts(
                error.normalized_form for error in errors
            ),
            confusion_counts=tuple(
                MorphologyFeatureConfusionCount(
                    gold_values=gold_values,
                    predicted_values=predicted_values,
                    count=count,
                )
                for (gold_values, predicted_values), count in _sorted_counts(
                    Counter(
                        (error.gold_values, error.predicted_values) for error in errors
                    )
                )
            ),
            context_counts=tuple(
                MorphologyContextCount(
                    previous_upos=previous_upos,
                    gold_upos=gold_upos,
                    next_upos=next_upos,
                    count=count,
                )
                for (previous_upos, gold_upos, next_upos), count in _sorted_counts(
                    Counter(
                        (
                            error.previous_gold_upos,
                            error.gold_upos,
                            error.next_gold_upos,
                        )
                        for error in errors
                    )
                )
            ),
            errors=errors,
        )


def _feature_values(
    universal_features: tuple[str, ...],
    *,
    feature_name: str,
) -> tuple[str, ...]:
    prefix = f"{feature_name}="
    for feature in universal_features:
        if feature.startswith(prefix):
            return tuple(feature.removeprefix(prefix).split(","))
    return ()


def _mapping_feature_values(value: str | None) -> tuple[str, ...]:
    return () if value is None else tuple(value.split(","))


def _named_counts(values: Iterable[str]) -> tuple[NamedCount, ...]:
    return tuple(
        NamedCount(name=name, count=count)
        for name, count in _sorted_counts(Counter(values))
    )


def _sorted_counts(
    counter: Counter[CountKey],
) -> tuple[tuple[CountKey, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
