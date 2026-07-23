"""Shared transformations for Norwegian UD treebanks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prism.conllu import Token
from prism.data.examples import (
    PretokenizedSentence,
    SupervisedCorpus,
    SupervisedSentence,
    TokenTargets,
)
from prism.schema import (
    TokenTaskSchema,
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
    derive_lemma_edit_rule,
    encode_morphology_targets,
)


def normalize_norwegian_ud_lemma(raw_lemma: str) -> str:
    return raw_lemma.removeprefix("$")


@dataclass(frozen=True, slots=True, kw_only=True)
class NorwegianUdLemmaDecoder:
    """Restore the treebank's ``$`` marker after internal normalization."""

    marker_token_forms: frozenset[str]

    def __call__(
        self,
        token_form: str,
        normalized_lemma: str,
        predicted_upos: str,
    ) -> str:
        del predicted_upos
        if token_form in self.marker_token_forms and not normalized_lemma.startswith(
            "$"
        ):
            return "$" + normalized_lemma
        return normalized_lemma


def build_norwegian_ud_lemma_decoder(
    training_sentences: Sequence[Sequence[Token]],
) -> NorwegianUdLemmaDecoder:
    return NorwegianUdLemmaDecoder(
        marker_token_forms=frozenset(
            token.text
            for sentence in training_sentences
            for token in sentence
            if token.lemma.startswith("$")
        )
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class NorwegianUdMorphologyDecoder:
    """Map canonical Norwegian morphology to one UD treebank convention."""

    language_tag: str

    def __post_init__(self) -> None:
        if self.language_tag not in ("nb", "nn"):
            raise ValueError(
                f"Unsupported Norwegian UD morphology profile: {self.language_tag}"
            )

    def __call__(
        self,
        predicted_upos: str,
        canonical_features: Mapping[str, str],
    ) -> dict[str, str]:
        decoded_features = self.decode_common_gender(
            predicted_upos,
            canonical_features,
        )

        if self.language_tag == "nn":
            decoded_features = self.decode_nynorsk_number(
                predicted_upos,
                decoded_features,
            )
            decoded_features = self.decode_nynorsk_definite(
                predicted_upos,
                decoded_features,
            )

        return decoded_features

    def decode_common_gender(
        self,
        predicted_upos: str,
        canonical_features: Mapping[str, str],
    ) -> dict[str, str]:
        decoded_features = dict(canonical_features)

        if (
            predicted_upos in ("ADJ", "DET")
            and decoded_features.get("Gender") == "Fem,Masc"
        ):
            decoded_features["Gender"] = "Com"

        return decoded_features

    def decode_nynorsk_number(
        self,
        predicted_upos: str,
        canonical_features: Mapping[str, str],
    ) -> dict[str, str]:
        del predicted_upos
        decoded_features = dict(canonical_features)
        if self.language_tag == "nn":
            _remove_morphology_value(
                decoded_features,
                feature_name="Number",
                value="Sing",
            )
        return decoded_features

    def decode_nynorsk_definite(
        self,
        predicted_upos: str,
        canonical_features: Mapping[str, str],
    ) -> dict[str, str]:
        del predicted_upos
        decoded_features = dict(canonical_features)
        if self.language_tag == "nn":
            _remove_morphology_value(
                decoded_features,
                feature_name="Definite",
                value="Def",
            )
        return decoded_features


def _remove_morphology_value(
    features: dict[str, str],
    *,
    feature_name: str,
    value: str,
) -> None:
    raw_values = features.get(feature_name)
    if raw_values is None:
        return

    remaining_values = tuple(
        feature_value
        for feature_value in raw_values.split(",")
        if feature_value != value
    )
    if remaining_values:
        features[feature_name] = ",".join(remaining_values)
    else:
        del features[feature_name]


def encode_norwegian_sentence(
    tokens: Sequence[Token],
    *,
    schema: TokenTaskSchema,
) -> SupervisedSentence:
    token_texts: list[str] = []
    has_space_before: list[bool] = []
    token_targets: list[TokenTargets] = []
    previous_token_has_space_after = False

    for token in tokens:
        has_space_before.append(previous_token_has_space_after)
        upos_id = schema.upos.label_id_for(token.upos)
        if upos_id is None:
            raise ValueError(f"Unknown UPOS label: {token.upos}")

        if token.lemma == "_":
            lemma_is_annotated = False
            lemma_rule_id = None
        else:
            lemma_is_annotated = True
            normalized_lemma = normalize_norwegian_ud_lemma(token.lemma)
            lemma_rule = derive_lemma_edit_rule(
                token.text,
                normalized_lemma,
            )
            lemma_rule_id = schema.lemma_rules.rule_id_for(lemma_rule)

        token_texts.append(token.text)
        token_targets.append(
            TokenTargets(
                upos_id=upos_id,
                morphology=encode_morphology_targets(
                    schema.morphology,
                    token.features,
                ),
                lemma_is_annotated=lemma_is_annotated,
                lemma_rule_id=lemma_rule_id,
            )
        )

        previous_token_has_space_after = token.space_after

    return SupervisedSentence(
        model_input=PretokenizedSentence(
            tokens=tuple(token_texts),
            has_space_before=tuple(has_space_before),
        ),
        targets=tuple(token_targets),
    )


def encode_norwegian_sentences(
    sentences: Sequence[Sequence[Token]],
    *,
    schema: TokenTaskSchema,
) -> SupervisedCorpus:
    return SupervisedCorpus(
        sentences=tuple(
            encode_norwegian_sentence(sentence, schema=schema) for sentence in sentences
        )
    )


def build_norwegian_schema(
    sentences: Sequence[Sequence[Token]],
) -> TokenTaskSchema:
    upos_schema = build_upos_schema(
        token.upos for sentence in sentences for token in sentence
    )
    morphology_schema = build_morphology_schema(
        token.features for sentence in sentences for token in sentence
    )
    lemma_rule_schema = build_lemma_rule_schema(
        (
            token.text,
            normalize_norwegian_ud_lemma(token.lemma),
        )
        for sentence in sentences
        for token in sentence
        if token.lemma != "_"
    )

    return TokenTaskSchema(
        upos=upos_schema,
        morphology=morphology_schema,
        lemma_rules=lemma_rule_schema,
    )
