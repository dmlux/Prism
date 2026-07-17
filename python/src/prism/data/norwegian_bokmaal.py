"""Norwegian Bokmål dataset-specific transformations."""

from collections.abc import Sequence

from prism.conllu import Token
from prism.data.examples import (
    PretokenizedSentence,
    SupervisedSentence,
    SupervisedCorpus,
    TokenTargets,
)
from prism.schema import (
    TokenTaskSchema,
    derive_lemma_edit_rule,
    encode_morphology_targets,
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
)


def normalize_norwegian_bokmaal_ud_lemma(
    raw_lemma: str,
) -> str:
    return raw_lemma.removeprefix("$")


def encode_norwegian_bokmaal_sentence(
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
            normalized_lemma = normalize_norwegian_bokmaal_ud_lemma(token.lemma)
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


def encode_norwegian_bokmaal_sentences(
    sentences: Sequence[Sequence[Token]],
    *,
    schema: TokenTaskSchema,
) -> SupervisedCorpus:
    return SupervisedCorpus(
        sentences=tuple(
            encode_norwegian_bokmaal_sentence(sentence, schema=schema)
            for sentence in sentences
        )
    )


def build_norwegian_bokmaal_schema(
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
            normalize_norwegian_bokmaal_ud_lemma(token.lemma),
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
