from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PretokenizedSentence:
    tokens: tuple[str, ...]
    has_space_before: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("Pretokenized sentence must contain tokens.")
        if len(self.tokens) != len(self.has_space_before):
            raise ValueError("Token and spacing counts must match.")
        if self.has_space_before[0]:
            raise ValueError("The first token cannot have preceding whitespace.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTargets:
    upos_id: int
    morphology: tuple[tuple[bool, ...], ...]
    lemma_is_annotated: bool
    lemma_rule_id: int | None

    def __post_init__(self) -> None:
        if self.upos_id < 0:
            raise ValueError("UPOS target ID must not be negative.")
        if not self.morphology:
            raise ValueError("Morphology targets must not be empty.")
        if any(not labels or not any(labels) for labels in self.morphology):
            raise ValueError(
                "Every morphology feature must activate at least one label."
            )
        if self.lemma_rule_id is not None and self.lemma_rule_id < 0:
            raise ValueError("Lemma rule ID must not be negative.")
        if not self.lemma_is_annotated and self.lemma_rule_id is not None:
            raise ValueError("A missing lemma annotation cannot have a lemma rule ID.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedSentence:
    model_input: PretokenizedSentence
    targets: tuple[TokenTargets, ...]

    def __post_init__(self) -> None:
        if len(self.model_input.tokens) != len(self.targets):
            raise ValueError("Token and targets counts must match.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedCorpus:
    sentences: tuple[SupervisedSentence, ...]

    def __post_init__(self) -> None:
        if not self.sentences:
            raise ValueError("Supervised corpus must contain sentences.")

    @property
    def token_count(self) -> int:
        return sum(len(sentence.model_input.tokens) for sentence in self.sentences)

    @property
    def lemma_annotation_count(self) -> int:
        return sum(
            target.lemma_is_annotated
            for sentence in self.sentences
            for target in sentence.targets
        )

    @property
    def unknown_lemma_rule_count(self) -> int:
        return sum(
            target.lemma_is_annotated and target.lemma_rule_id is None
            for sentence in self.sentences
            for target in sentence.targets
        )
