import math

import torch

from prism.conllu import Token
from prism.evaluation import (
    UniversalDependenciesEvaluationAccumulator,
    UniversalFeaturesPolicyStep,
    build_universal_dependencies_reference_batch,
    evaluate_gold_tokenized_conllu,
)
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.schema import (
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
    derive_lemma_edit_rule,
    TokenTaskSchema,
)


def test_gold_tokenized_ud_metrics_match_official_score_semantics() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(("NOUN", "PUNCT")),
        morphology=build_morphology_schema(
            (
                {"Gender": "Neut", "Number": "Sing"},
                {},
            )
        ),
        lemma_rules=build_lemma_rule_schema(
            (
                ("Huset", "hus"),
                (".", "."),
            )
        ),
    )
    references = (
        build_universal_dependencies_reference_batch(
            (
                (
                    Token(
                        text="Huset",
                        lemma="hus",
                        upos="NOUN",
                        features={"Gender": "Neut", "Number": "Sing"},
                    ),
                    Token(
                        text=".",
                        lemma="$.",
                        upos="PUNCT",
                        features={},
                    ),
                ),
            )
        ),
    )
    accumulator = UniversalDependenciesEvaluationAccumulator(
        schema=schema,
        reference_batches=references,
        lemma_decoder=(lambda form, lemma, upos: "$" + lemma if form == "." else lemma),
    )
    predictions = TokenTaskPredictionBatch(
        upos_ids=torch.tensor(
            [[schema.upos.label_id_for("NOUN"), schema.upos.label_id_for("NOUN")]],
            dtype=torch.long,
        ),
        morphology_predictions=(
            torch.tensor(
                [[[False, True], [True, False]]],
                dtype=torch.bool,
            ),
            torch.tensor(
                [[[False, True], [True, False]]],
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor(
            [
                [
                    schema.lemma_rules.rule_id_for(
                        derive_lemma_edit_rule("Huset", "hus")
                    ),
                    schema.lemma_rules.rule_id_for(derive_lemma_edit_rule(".", ".")),
                ]
            ],
            dtype=torch.long,
        ),
        token_mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    accumulator.add(predictions=predictions)

    metrics = accumulator.finish()

    assert metrics.upos.gold_total == 2
    assert metrics.upos.correct == 1
    assert metrics.upos.precision == 0.5
    assert metrics.upos.recall == 0.5
    assert metrics.upos.f1 == 0.5
    assert metrics.upos.aligned_accuracy == 0.5
    assert metrics.ufeats.f1 == 1.0
    assert math.isclose(metrics.lemmas.f1, 1.0)

    token_slice_accumulator = accumulator.spawn_empty()
    token_slice_accumulator.add(
        predictions=predictions,
        evaluation_mask=torch.tensor([[False, True]], dtype=torch.bool),
    )
    token_slice_metrics = token_slice_accumulator.finish()

    assert token_slice_metrics.upos.gold_total == 1
    assert token_slice_metrics.upos.f1 == 0.0
    assert token_slice_metrics.ufeats.f1 == 1.0
    assert token_slice_metrics.lemmas.f1 == 1.0


def test_gold_tokenized_conllu_metrics_compare_complete_ufeats() -> None:
    gold = (
        (
            Token(
                text="hus",
                lemma="hus",
                upos="NOUN",
                features={"Gender": "Neut", "Number": "Sing"},
            ),
            Token(text="?", lemma="_", upos="PUNCT", features={}),
        ),
    )
    system = (
        (
            Token(
                text="hus",
                lemma="hus",
                upos="NOUN",
                features={"Gender": "Neut"},
            ),
            Token(text="?", lemma="anything", upos="NOUN", features={}),
        ),
    )

    metrics = evaluate_gold_tokenized_conllu(
        gold_sentences=gold,
        system_sentences=system,
    )

    assert metrics.upos.f1 == 0.5
    assert metrics.ufeats.f1 == 0.5
    assert metrics.lemmas.f1 == 1.0


def test_ud_accumulator_audits_universal_features_policy_steps() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(("ADJ",)),
        morphology=build_morphology_schema(({"Gender": "Com"}, {"Gender": "Masc"})),
        lemma_rules=build_lemma_rule_schema((("god", "god"),)),
    )
    references = (
        build_universal_dependencies_reference_batch(
            (
                (
                    Token(
                        text="god",
                        lemma="god",
                        upos="ADJ",
                        features={"Gender": "Com"},
                    ),
                ),
            )
        ),
    )
    accumulator = UniversalDependenciesEvaluationAccumulator(
        schema=schema,
        reference_batches=references,
        universal_features_policy_steps=(
            UniversalFeaturesPolicyStep(
                name="common-gender",
                decoder=(
                    lambda upos, features: (
                        {"Gender": "Com"}
                        if upos == "ADJ" and features == {"Gender": "Masc"}
                        else features
                    )
                ),
            ),
        ),
    )
    accumulator.add(
        predictions=TokenTaskPredictionBatch(
            upos_ids=torch.tensor([[schema.upos.label_id_for("ADJ")]]),
            morphology_predictions=(
                torch.tensor([[[False, False, True]]], dtype=torch.bool),
            ),
            lemma_rule_ids=torch.tensor(
                [[schema.lemma_rules.rule_id_for(derive_lemma_edit_rule("god", "god"))]]
            ),
            token_mask=torch.tensor([[True]]),
        )
    )

    metrics = accumulator.finish()

    assert metrics.ufeats.f1 == 1.0
    assert len(metrics.ufeats_policy_audits) == 1
    audit = metrics.ufeats_policy_audits[0]
    assert audit.name == "common-gender"
    assert audit.changed_bundle_count == 1
    assert audit.improved_bundle_count == 1
    assert audit.regressed_bundle_count == 0
