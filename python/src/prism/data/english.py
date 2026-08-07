"""Shared transformations for English UD treebanks.

Mirrors ``prism.data.norwegian`` so the English CLI package can delegate to
the same shared schema/encoding machinery. English needs far less
language-specific treatment than Norwegian:

* **Lemma normalization is the identity.** UD_English-EWT uses no ``$`` lemma
  marker; the ``$`` that appears there is the literal dollar-sign token (its
  lemma is ``$``), so stripping it — as the Norwegian rule does — would corrupt
  those lemmata. English lemmata are used verbatim to derive edit rules.
* **Morphology decoding is the identity.** EWT carries standard English UD
  features and none of the canonical→convention remappings Norwegian needs
  (no ``Gender=Fem,Masc``→``Com``, no Nynorsk number/definite trimming).
"""

from collections.abc import Mapping, Sequence

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


def normalize_english_ud_lemma(raw_lemma: str) -> str:
    """Return the treebank lemma unchanged (English needs no normalization)."""

    return raw_lemma


class EnglishUdLemmaDecoder:
    """Identity lemma decoder — English restores no normalization marker."""

    def __call__(
        self,
        token_form: str,
        normalized_lemma: str,
        predicted_upos: str,
    ) -> str:
        del token_form, predicted_upos
        return normalized_lemma


def build_english_ud_lemma_decoder(
    training_sentences: Sequence[Sequence[Token]],
) -> EnglishUdLemmaDecoder:
    del training_sentences
    return EnglishUdLemmaDecoder()


class EnglishUdMorphologyDecoder:
    """Identity morphology decoder — English UD features need no remapping."""

    def __call__(
        self,
        predicted_upos: str,
        canonical_features: Mapping[str, str],
    ) -> dict[str, str]:
        del predicted_upos
        return dict(canonical_features)


def encode_english_sentence(
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
            normalized_lemma = normalize_english_ud_lemma(token.lemma)
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


def encode_english_sentences(
    sentences: Sequence[Sequence[Token]],
    *,
    schema: TokenTaskSchema,
) -> SupervisedCorpus:
    return SupervisedCorpus(
        sentences=tuple(
            encode_english_sentence(sentence, schema=schema) for sentence in sentences
        )
    )


def build_english_schema(
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
            normalize_english_ud_lemma(token.lemma),
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
