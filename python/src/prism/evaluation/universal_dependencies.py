"""Gold-tokenized metrics compatible with the official UD evaluator."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from prism.conllu import Token
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.schema import TokenTaskSchema, decode_morphology_values


UNIVERSAL_FEATURE_NAMES = frozenset(
    {
        "Abbr",
        "Animacy",
        "Aspect",
        "Case",
        "Definite",
        "Degree",
        "Evident",
        "Foreign",
        "Gender",
        "Mood",
        "NumType",
        "Number",
        "Person",
        "Polarity",
        "Polite",
        "Poss",
        "PronType",
        "Reflex",
        "Tense",
        "VerbForm",
        "Voice",
    }
)

LemmaDecoder = Callable[[str, str, str], str]
UniversalFeaturesDecoder = Callable[
    [str, Mapping[str, str]],
    Mapping[str, str],
]


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalFeaturesPolicyStep:
    name: str
    decoder: UniversalFeaturesDecoder

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError(
                "UD feature-policy step name must be non-empty and trimmed."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalFeaturesPolicyAudit:
    name: str
    changed_bundle_count: int
    improved_bundle_count: int
    regressed_bundle_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalDependenciesTokenReference:
    form: str
    lemma: str
    upos: str
    universal_features: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalDependenciesReferenceBatch:
    sentences: tuple[tuple[UniversalDependenciesTokenReference, ...], ...]

    def __post_init__(self) -> None:
        if not self.sentences:
            raise ValueError("UD reference batch must contain sentences.")
        if any(not sentence for sentence in self.sentences):
            raise ValueError("UD reference sentences must contain tokens.")


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalDependenciesMetricScore:
    gold_total: int
    system_total: int
    correct: int
    aligned_total: int

    @property
    def precision(self) -> float:
        return self.correct / self.system_total if self.system_total else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold_total if self.gold_total else 0.0

    @property
    def f1(self) -> float:
        total = self.system_total + self.gold_total
        return 2 * self.correct / total if total else 0.0

    @property
    def aligned_accuracy(self) -> float | None:
        if not self.aligned_total:
            return None
        return self.correct / self.aligned_total


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalDependenciesEvaluationMetrics:
    upos: UniversalDependenciesMetricScore
    ufeats: UniversalDependenciesMetricScore
    lemmas: UniversalDependenciesMetricScore
    ufeats_policy_audits: tuple[UniversalFeaturesPolicyAudit, ...] = ()


def serialize_universal_dependencies_evaluation_metrics(
    metrics: UniversalDependenciesEvaluationMetrics,
) -> dict[str, object]:
    def serialize_score(
        score: UniversalDependenciesMetricScore,
    ) -> dict[str, int | float | None]:
        return {
            "gold_total": score.gold_total,
            "system_total": score.system_total,
            "correct": score.correct,
            "aligned_total": score.aligned_total,
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "aligned_accuracy": score.aligned_accuracy,
        }

    serialized: dict[str, object] = {
        "UPOS": serialize_score(metrics.upos),
        "UFeats": serialize_score(metrics.ufeats),
        "Lemmas": serialize_score(metrics.lemmas),
    }
    if metrics.ufeats_policy_audits:
        serialized["UFeatsPolicyAudits"] = [
            {
                "name": audit.name,
                "changed_bundle_count": audit.changed_bundle_count,
                "improved_bundle_count": audit.improved_bundle_count,
                "regressed_bundle_count": audit.regressed_bundle_count,
            }
            for audit in metrics.ufeats_policy_audits
        ]
    return serialized


def build_universal_dependencies_reference_batch(
    sentences: Sequence[Sequence[Token]],
) -> UniversalDependenciesReferenceBatch:
    return UniversalDependenciesReferenceBatch(
        sentences=tuple(
            tuple(
                UniversalDependenciesTokenReference(
                    form=token.text,
                    lemma=token.lemma,
                    upos=token.upos,
                    universal_features=tuple(
                        sorted(
                            f"{name}={value}"
                            for name, value in token.features.items()
                            if name in UNIVERSAL_FEATURE_NAMES
                        )
                    ),
                )
                for token in sentence
            )
            for sentence in sentences
        )
    )


def evaluate_gold_tokenized_conllu(
    *,
    gold_sentences: Sequence[Sequence[Token]],
    system_sentences: Sequence[Sequence[Token]],
) -> UniversalDependenciesEvaluationMetrics:
    """Evaluate already aligned CoNLL-U words like the official UD scorer."""

    if len(gold_sentences) != len(system_sentences):
        raise ValueError("Gold and system CoNLL-U sentence counts must match.")

    token_count = 0
    upos_correct_count = 0
    ufeats_correct_count = 0
    lemma_correct_count = 0

    for gold_sentence, system_sentence in zip(
        gold_sentences,
        system_sentences,
        strict=True,
    ):
        if len(gold_sentence) != len(system_sentence):
            raise ValueError("Gold and system CoNLL-U token counts must match.")

        for gold, system in zip(gold_sentence, system_sentence, strict=True):
            if gold.text != system.text:
                raise ValueError(
                    "Gold-tokenized evaluation requires identical token forms."
                )

            gold_features = tuple(
                sorted(
                    f"{name}={value}"
                    for name, value in gold.features.items()
                    if name in UNIVERSAL_FEATURE_NAMES
                )
            )
            system_features = tuple(
                sorted(
                    f"{name}={value}"
                    for name, value in system.features.items()
                    if name in UNIVERSAL_FEATURE_NAMES
                )
            )

            token_count += 1
            upos_correct_count += gold.upos == system.upos
            ufeats_correct_count += gold_features == system_features
            lemma_correct_count += gold.lemma == "_" or gold.lemma == system.lemma

    if not token_count:
        raise ValueError("Gold-tokenized UD evaluation must contain tokens.")

    def score(correct: int) -> UniversalDependenciesMetricScore:
        return UniversalDependenciesMetricScore(
            gold_total=token_count,
            system_total=token_count,
            correct=correct,
            aligned_total=token_count,
        )

    return UniversalDependenciesEvaluationMetrics(
        upos=score(upos_correct_count),
        ufeats=score(ufeats_correct_count),
        lemmas=score(lemma_correct_count),
    )


@dataclass(slots=True, kw_only=True)
class UniversalDependenciesEvaluationAccumulator:
    schema: TokenTaskSchema
    reference_batches: tuple[UniversalDependenciesReferenceBatch, ...]
    lemma_decoder: LemmaDecoder | None = None
    universal_features_policy_steps: tuple[UniversalFeaturesPolicyStep, ...] = ()
    _batch_index: int = 0
    _token_count: int = 0
    _upos_correct_count: int = 0
    _ufeats_correct_count: int = 0
    _lemma_correct_count: int = 0
    _policy_changed_counts: list[int] = field(init=False)
    _policy_improved_counts: list[int] = field(init=False)
    _policy_regressed_counts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        step_names = tuple(step.name for step in self.universal_features_policy_steps)
        if len(set(step_names)) != len(step_names):
            raise ValueError("UD feature-policy step names must be unique.")
        self._policy_changed_counts = [0] * len(step_names)
        self._policy_improved_counts = [0] * len(step_names)
        self._policy_regressed_counts = [0] * len(step_names)

    def add(self, *, predictions: TokenTaskPredictionBatch) -> None:
        if self._batch_index >= len(self.reference_batches):
            raise ValueError("UD evaluation received more predictions than references.")

        reference_batch = self.reference_batches[self._batch_index]
        if predictions.upos_ids.shape[0] != len(reference_batch.sentences):
            raise ValueError("UD prediction and reference batch sizes must match.")

        for sentence_index, reference_sentence in enumerate(reference_batch.sentences):
            predicted_token_count = int(
                predictions.token_mask[sentence_index].sum().item()
            )
            if predicted_token_count != len(reference_sentence):
                raise ValueError(
                    "UD prediction and reference sentence token counts must match."
                )

            for token_index, reference in enumerate(reference_sentence):
                predicted_upos = self.schema.upos.label_for_id(
                    int(predictions.upos_ids[sentence_index, token_index].item())
                )
                predicted_features = decode_morphology_values(
                    self.schema.morphology,
                    tuple(
                        tuple(
                            bool(value)
                            for value in feature_predictions[
                                sentence_index, token_index
                            ].tolist()
                        )
                        for feature_predictions in predictions.morphology_predictions
                    ),
                )
                for step_index, policy_step in enumerate(
                    self.universal_features_policy_steps
                ):
                    before_features = tuple(
                        sorted(
                            f"{name}={value}"
                            for name, value in predicted_features.items()
                            if name in UNIVERSAL_FEATURE_NAMES
                        )
                    )
                    predicted_features = dict(
                        policy_step.decoder(
                            predicted_upos,
                            predicted_features,
                        )
                    )
                    after_features = tuple(
                        sorted(
                            f"{name}={value}"
                            for name, value in predicted_features.items()
                            if name in UNIVERSAL_FEATURE_NAMES
                        )
                    )
                    if before_features != after_features:
                        self._policy_changed_counts[step_index] += 1
                    if (
                        before_features != reference.universal_features
                        and after_features == reference.universal_features
                    ):
                        self._policy_improved_counts[step_index] += 1
                    if (
                        before_features == reference.universal_features
                        and after_features != reference.universal_features
                    ):
                        self._policy_regressed_counts[step_index] += 1
                predicted_universal_features = tuple(
                    sorted(
                        f"{name}={value}"
                        for name, value in predicted_features.items()
                        if name in UNIVERSAL_FEATURE_NAMES
                    )
                )
                predicted_lemma_rule = self.schema.lemma_rules.rule_for_id(
                    int(predictions.lemma_rule_ids[sentence_index, token_index].item())
                )
                try:
                    predicted_lemma = predicted_lemma_rule.apply(reference.form)
                except ValueError:
                    # A globally shared edit-rule inventory can select a rule that
                    # removes more characters than a short token contains. It is an
                    # incorrect lemma prediction, not an evaluation failure.
                    predicted_lemma = "<INVALID_LEMMA_RULE>"
                if self.lemma_decoder is not None:
                    predicted_lemma = self.lemma_decoder(
                        reference.form,
                        predicted_lemma,
                        predicted_upos,
                    )

                self._token_count += 1
                self._upos_correct_count += predicted_upos == reference.upos
                self._ufeats_correct_count += (
                    predicted_universal_features == reference.universal_features
                )
                # The official scorer ignores system lemmas where gold is '_'.
                self._lemma_correct_count += (
                    reference.lemma == "_" or predicted_lemma == reference.lemma
                )

        self._batch_index += 1

    def finish(self) -> UniversalDependenciesEvaluationMetrics:
        if self._batch_index != len(self.reference_batches):
            raise ValueError("UD evaluation contains unused reference batches.")
        if not self._token_count:
            raise ValueError("UD evaluation must contain tokens.")

        def score(correct: int) -> UniversalDependenciesMetricScore:
            return UniversalDependenciesMetricScore(
                gold_total=self._token_count,
                system_total=self._token_count,
                correct=correct,
                aligned_total=self._token_count,
            )

        return UniversalDependenciesEvaluationMetrics(
            upos=score(self._upos_correct_count),
            ufeats=score(self._ufeats_correct_count),
            lemmas=score(self._lemma_correct_count),
            ufeats_policy_audits=tuple(
                UniversalFeaturesPolicyAudit(
                    name=step.name,
                    changed_bundle_count=changed,
                    improved_bundle_count=improved,
                    regressed_bundle_count=regressed,
                )
                for step, changed, improved, regressed in zip(
                    self.universal_features_policy_steps,
                    self._policy_changed_counts,
                    self._policy_improved_counts,
                    self._policy_regressed_counts,
                    strict=True,
                )
            ),
        )
