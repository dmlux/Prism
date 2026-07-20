from prism.conllu import Token
from prism.data.norwegian import (
    encode_norwegian_sentence,
    encode_norwegian_sentences,
)
from prism.schema import (
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
    TokenTaskSchema,
)
from prism.data import PretokenizedSentence


def test_encode_norwegian_sentence_builds_targets() -> None:
    upos_schema = build_upos_schema(
        [
            "NOUN",
            "PUNCT",
        ]
    )
    morphology_schema = build_morphology_schema(
        [
            {"Number": "Plur"},
            {},
        ]
    )
    lemma_rule_schema = build_lemma_rule_schema(
        [
            ("husene", "hus"),
            (".", "."),
        ]
    )
    schema = TokenTaskSchema(
        upos=upos_schema,
        morphology=morphology_schema,
        lemma_rules=lemma_rule_schema,
    )

    sentence = encode_norwegian_sentence(
        [
            Token(
                text="husene",
                lemma="hus",
                upos="NOUN",
                features={"Number": "Plur"},
                space_after=False,
            ),
            Token(
                text=".",
                lemma="$.",
                upos="PUNCT",
                features={},
            ),
        ],
        schema=schema,
    )

    assert sentence.model_input == PretokenizedSentence(
        tokens=("husene", "."),
        has_space_before=(False, False),
    )
    assert sentence.targets[0].upos_id == 0
    assert sentence.targets[0].morphology == ((False, True),)
    assert sentence.targets[0].lemma_rule_id == 1

    assert sentence.targets[1].upos_id == 1
    assert sentence.targets[1].morphology == ((True, False),)
    assert sentence.targets[1].lemma_rule_id == 0


def test_encode_sentence_distinguishes_missing_and_unknown_lemmas() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(
            [
                "VERB",
                "X",
            ]
        ),
        morphology=build_morphology_schema(
            [
                {"Number": "Sing"},
            ]
        ),
        lemma_rules=build_lemma_rule_schema(
            [
                ("hus", "hus"),
            ]
        ),
    )

    sentence = encode_norwegian_sentence(
        [
            Token(
                text="gikk",
                lemma="gå",
                upos="VERB",
                features={},
            ),
            Token(
                text="ukjent",
                lemma="_",
                upos="X",
                features={},
            ),
        ],
        schema=schema,
    )

    unknown_rule_target = sentence.targets[0]
    assert unknown_rule_target.lemma_is_annotated is True
    assert unknown_rule_target.lemma_rule_id is None

    missing_lemma_target = sentence.targets[1]
    assert missing_lemma_target.lemma_is_annotated is False
    assert missing_lemma_target.lemma_rule_id is None


def test_encode_sentences_reports_lemma_rule_coverage() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(
            [
                "NOUN",
                "VERB",
            ]
        ),
        morphology=build_morphology_schema(
            [
                {"Number": "Sing"},
            ]
        ),
        lemma_rules=build_lemma_rule_schema(
            [
                ("hus", "hus"),
            ]
        ),
    )

    corpus = encode_norwegian_sentences(
        [
            [
                Token(
                    text="hus",
                    lemma="hus",
                    upos="NOUN",
                    features={},
                ),
            ],
            [
                Token(
                    text="gikk",
                    lemma="gå",
                    upos="VERB",
                    features={},
                ),
            ],
        ],
        schema=schema,
    )

    assert len(corpus.sentences) == 2
    assert corpus.token_count == 2
    assert corpus.lemma_annotation_count == 2
    assert corpus.unknown_lemma_rule_count == 1
