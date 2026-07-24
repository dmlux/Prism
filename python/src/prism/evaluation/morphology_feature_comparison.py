"""Token-aligned feature-level morphology comparison for fixed UD splits."""

from collections import Counter
from dataclasses import dataclass, field

from prism.evaluation.token_frequency import (
    TokenFrequencyClass,
    TokenFrequencyProfile,
)
from prism.evaluation.universal_dependencies import (
    UNIVERSAL_FEATURE_NAMES,
    UniversalDependenciesReferenceBatch,
    UniversalFeaturesPolicyStep,
)
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.schema import TokenTaskSchema, decode_morphology_values


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureValueMetrics:
    value: str
    support: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive_count + self.false_positive_count
        return self.true_positive_count / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive_count + self.false_negative_count
        return self.true_positive_count / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = (
            2 * self.true_positive_count
            + self.false_positive_count
            + self.false_negative_count
        )
        return 2 * self.true_positive_count / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureSystemMetrics:
    token_count: int
    correct_count: int
    annotated_token_count: int
    annotated_correct_count: int
    wrong_bundle_count: int
    feature_error_in_wrong_bundle_count: int
    values: tuple[MorphologyFeatureValueMetrics, ...]

    @property
    def overall_accuracy(self) -> float:
        return self.correct_count / self.token_count if self.token_count else 0.0

    @property
    def annotated_accuracy(self) -> float | None:
        if not self.annotated_token_count:
            return None
        return self.annotated_correct_count / self.annotated_token_count

    @property
    def wrong_bundle_error_share(self) -> float | None:
        if not self.wrong_bundle_count:
            return None
        return self.feature_error_in_wrong_bundle_count / self.wrong_bundle_count


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureSliceComparison:
    name: str
    model: MorphologyFeatureSystemMetrics
    comparison: MorphologyFeatureSystemMetrics


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureComparison:
    feature_name: str
    model: MorphologyFeatureSystemMetrics
    comparison: MorphologyFeatureSystemMetrics
    frequency_slices: tuple[MorphologyFeatureSliceComparison, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureComparisonReport:
    token_count: int
    features: tuple[MorphologyFeatureComparison, ...]


@dataclass(slots=True)
class _MutableFeatureCounts:
    token_count: int = 0
    correct_count: int = 0
    annotated_token_count: int = 0
    annotated_correct_count: int = 0
    wrong_bundle_count: int = 0
    feature_error_in_wrong_bundle_count: int = 0
    support_counts: Counter[str] = field(default_factory=Counter)
    true_positive_counts: Counter[str] = field(default_factory=Counter)
    false_positive_counts: Counter[str] = field(default_factory=Counter)
    false_negative_counts: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        *,
        gold_values: tuple[str, ...],
        predicted_values: tuple[str, ...],
        bundle_correct: bool,
    ) -> None:
        gold = frozenset(gold_values)
        predicted = frozenset(predicted_values)
        feature_correct = gold == predicted

        self.token_count += 1
        self.correct_count += feature_correct
        if gold:
            self.annotated_token_count += 1
            self.annotated_correct_count += feature_correct
        if not bundle_correct:
            self.wrong_bundle_count += 1
            self.feature_error_in_wrong_bundle_count += not feature_correct

        for value in gold:
            self.support_counts[value] += 1
        for value in gold & predicted:
            self.true_positive_counts[value] += 1
        for value in predicted - gold:
            self.false_positive_counts[value] += 1
        for value in gold - predicted:
            self.false_negative_counts[value] += 1

    def finish(self, *, values: tuple[str, ...]) -> MorphologyFeatureSystemMetrics:
        return MorphologyFeatureSystemMetrics(
            token_count=self.token_count,
            correct_count=self.correct_count,
            annotated_token_count=self.annotated_token_count,
            annotated_correct_count=self.annotated_correct_count,
            wrong_bundle_count=self.wrong_bundle_count,
            feature_error_in_wrong_bundle_count=(
                self.feature_error_in_wrong_bundle_count
            ),
            values=tuple(
                MorphologyFeatureValueMetrics(
                    value=value,
                    support=self.support_counts[value],
                    true_positive_count=self.true_positive_counts[value],
                    false_positive_count=self.false_positive_counts[value],
                    false_negative_count=self.false_negative_counts[value],
                )
                for value in values
            ),
        )


@dataclass(slots=True, kw_only=True)
class MorphologyFeatureComparisonAccumulator:
    schema: TokenTaskSchema
    reference_batches: tuple[UniversalDependenciesReferenceBatch, ...]
    comparison_reference_batches: tuple[UniversalDependenciesReferenceBatch, ...]
    frequency_profile: TokenFrequencyProfile
    universal_features_policy_steps: tuple[UniversalFeaturesPolicyStep, ...] = ()
    _batch_index: int = 0
    _token_count: int = 0
    _feature_names: tuple[str, ...] = field(init=False)
    _feature_values: dict[str, tuple[str, ...]] = field(init=False)
    _model_counts: dict[str, _MutableFeatureCounts] = field(init=False)
    _comparison_counts: dict[str, _MutableFeatureCounts] = field(init=False)
    _slice_model_counts: dict[TokenFrequencyClass, dict[str, _MutableFeatureCounts]] = (
        field(init=False)
    )
    _slice_comparison_counts: dict[
        TokenFrequencyClass, dict[str, _MutableFeatureCounts]
    ] = field(init=False)

    def __post_init__(self) -> None:
        if not self.reference_batches:
            raise ValueError("Morphology feature comparison requires references.")
        if len(self.reference_batches) != len(self.comparison_reference_batches):
            raise ValueError("Morphology feature comparison batch counts must match.")

        self._feature_names = tuple(
            feature.name
            for feature in self.schema.morphology.features
            if feature.name in UNIVERSAL_FEATURE_NAMES
        )
        self._feature_values = {
            feature.name: feature.values
            for feature in self.schema.morphology.features
            if feature.name in UNIVERSAL_FEATURE_NAMES
        }
        self._model_counts = self._new_feature_counts()
        self._comparison_counts = self._new_feature_counts()
        slice_classes = (TokenFrequencyClass.RARE, TokenFrequencyClass.OOV)
        self._slice_model_counts = {
            frequency_class: self._new_feature_counts()
            for frequency_class in slice_classes
        }
        self._slice_comparison_counts = {
            frequency_class: self._new_feature_counts()
            for frequency_class in slice_classes
        }

    def _new_feature_counts(self) -> dict[str, _MutableFeatureCounts]:
        return {
            feature_name: _MutableFeatureCounts()
            for feature_name in self._feature_names
        }

    def add(self, *, predictions: TokenTaskPredictionBatch) -> None:
        if self._batch_index >= len(self.reference_batches):
            raise ValueError(
                "Morphology feature comparison received too many predictions."
            )

        reference_batch = self.reference_batches[self._batch_index]
        comparison_batch = self.comparison_reference_batches[self._batch_index]
        if predictions.upos_ids.shape[0] != len(reference_batch.sentences):
            raise ValueError(
                "Morphology feature comparison prediction and reference batch "
                "sizes must match."
            )
        if len(reference_batch.sentences) != len(comparison_batch.sentences):
            raise ValueError(
                "Morphology feature comparison sentence counts must match."
            )

        for sentence_index, (reference_sentence, comparison_sentence) in enumerate(
            zip(
                reference_batch.sentences,
                comparison_batch.sentences,
                strict=True,
            )
        ):
            predicted_token_count = int(
                predictions.token_mask[sentence_index].sum().item()
            )
            if predicted_token_count != len(reference_sentence):
                raise ValueError(
                    "Morphology feature comparison prediction and reference token "
                    "counts must match."
                )
            if len(reference_sentence) != len(comparison_sentence):
                raise ValueError(
                    "Morphology feature comparison token counts must match."
                )

            for token_index, (reference, comparison) in enumerate(
                zip(reference_sentence, comparison_sentence, strict=True)
            ):
                if reference.form != comparison.form:
                    raise ValueError(
                        "Morphology feature comparison requires identical token forms."
                    )

                predicted_upos = self.schema.upos.label_for_id(
                    int(predictions.upos_ids[sentence_index, token_index].item())
                )
                model_features = dict(
                    decode_morphology_values(
                        self.schema.morphology,
                        tuple(
                            tuple(
                                bool(value)
                                for value in feature_predictions[
                                    sentence_index, token_index
                                ].tolist()
                            )
                            for feature_predictions in (
                                predictions.morphology_predictions
                            )
                        ),
                    )
                )
                for policy_step in self.universal_features_policy_steps:
                    model_features = dict(
                        policy_step.decoder(predicted_upos, model_features)
                    )

                gold_features = _universal_feature_mapping(reference.universal_features)
                model_universal_features = _mapping_to_universal_features(
                    model_features
                )
                comparison_features = _universal_feature_mapping(
                    comparison.universal_features
                )
                model_bundle_correct = model_universal_features == gold_features
                comparison_bundle_correct = comparison_features == gold_features
                frequency_class = self.frequency_profile.classify(reference.form)

                for feature_name in self._feature_names:
                    gold_values = gold_features.get(feature_name, ())
                    model_values = model_universal_features.get(feature_name, ())
                    comparison_values = comparison_features.get(feature_name, ())
                    self._model_counts[feature_name].add(
                        gold_values=gold_values,
                        predicted_values=model_values,
                        bundle_correct=model_bundle_correct,
                    )
                    self._comparison_counts[feature_name].add(
                        gold_values=gold_values,
                        predicted_values=comparison_values,
                        bundle_correct=comparison_bundle_correct,
                    )
                    if frequency_class in self._slice_model_counts:
                        self._slice_model_counts[frequency_class][feature_name].add(
                            gold_values=gold_values,
                            predicted_values=model_values,
                            bundle_correct=model_bundle_correct,
                        )
                        self._slice_comparison_counts[frequency_class][
                            feature_name
                        ].add(
                            gold_values=gold_values,
                            predicted_values=comparison_values,
                            bundle_correct=comparison_bundle_correct,
                        )

                self._token_count += 1

        self._batch_index += 1

    def finish(self) -> MorphologyFeatureComparisonReport:
        if self._batch_index != len(self.reference_batches):
            raise ValueError(
                "Morphology feature comparison contains unused references."
            )
        if not self._token_count:
            raise ValueError("Morphology feature comparison must contain tokens.")

        return MorphologyFeatureComparisonReport(
            token_count=self._token_count,
            features=tuple(
                MorphologyFeatureComparison(
                    feature_name=feature_name,
                    model=self._model_counts[feature_name].finish(
                        values=self._feature_values[feature_name]
                    ),
                    comparison=self._comparison_counts[feature_name].finish(
                        values=self._feature_values[feature_name]
                    ),
                    frequency_slices=tuple(
                        MorphologyFeatureSliceComparison(
                            name=frequency_class.value,
                            model=self._slice_model_counts[frequency_class][
                                feature_name
                            ].finish(values=self._feature_values[feature_name]),
                            comparison=self._slice_comparison_counts[frequency_class][
                                feature_name
                            ].finish(values=self._feature_values[feature_name]),
                        )
                        for frequency_class in (
                            TokenFrequencyClass.RARE,
                            TokenFrequencyClass.OOV,
                        )
                    ),
                )
                for feature_name in self._feature_names
            ),
        )


def serialize_morphology_feature_comparison_report(
    report: MorphologyFeatureComparisonReport,
) -> dict[str, object]:
    def serialize_system(
        metrics: MorphologyFeatureSystemMetrics,
    ) -> dict[str, object]:
        return {
            "token_count": metrics.token_count,
            "correct_count": metrics.correct_count,
            "overall_accuracy": metrics.overall_accuracy,
            "annotated_token_count": metrics.annotated_token_count,
            "annotated_correct_count": metrics.annotated_correct_count,
            "annotated_accuracy": metrics.annotated_accuracy,
            "wrong_bundle_count": metrics.wrong_bundle_count,
            "feature_error_in_wrong_bundle_count": (
                metrics.feature_error_in_wrong_bundle_count
            ),
            "wrong_bundle_error_share": metrics.wrong_bundle_error_share,
            "values": [
                {
                    "value": value.value,
                    "support": value.support,
                    "true_positive_count": value.true_positive_count,
                    "false_positive_count": value.false_positive_count,
                    "false_negative_count": value.false_negative_count,
                    "precision": value.precision,
                    "recall": value.recall,
                    "f1": value.f1,
                }
                for value in metrics.values
            ],
        }

    return {
        "token_count": report.token_count,
        "features": [
            {
                "feature_name": feature.feature_name,
                "model": serialize_system(feature.model),
                "comparison": serialize_system(feature.comparison),
                "frequency_slices": [
                    {
                        "name": token_slice.name,
                        "model": serialize_system(token_slice.model),
                        "comparison": serialize_system(token_slice.comparison),
                    }
                    for token_slice in feature.frequency_slices
                ],
            }
            for feature in report.features
        ],
    }


def _universal_feature_mapping(
    universal_features: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for feature in universal_features:
        name, values = feature.split("=", 1)
        if name in UNIVERSAL_FEATURE_NAMES:
            mapping[name] = tuple(sorted(values.split(",")))
    return mapping


def _mapping_to_universal_features(
    features: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(sorted(value.split(",")))
        for name, value in features.items()
        if name in UNIVERSAL_FEATURE_NAMES
    }
