import torch

from prism.conllu import Token
from prism.evaluation import (
    MorphologyErrorAuditAccumulator,
    TokenFrequencyProfile,
    build_universal_dependencies_reference_batch,
)
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.schema import (
    NO_MORPHOLOGY_VALUE,
    TokenTaskSchema,
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
)


def test_morphology_error_audit_attributes_errors_and_comparison_success() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(("ADJ", "DET", "NOUN")),
        morphology=build_morphology_schema(
            (
                {"Gender": "Masc", "Number": "Sing"},
                {"Gender": "Fem", "Number": "Sing"},
                {"Gender": "Com", "Number": "Sing"},
                {"Gender": "Neut", "Number": "Sing"},
            )
        ),
        lemma_rules=build_lemma_rule_schema(
            (("den", "den"), ("boka", "bok"), ("x", "x"), ("hus", "hus"))
        ),
    )
    gold_tokens = (
        (
            Token(
                text="den",
                lemma="den",
                upos="DET",
                features={"Gender": "Masc", "Number": "Sing"},
            ),
            Token(
                text="boka",
                lemma="bok",
                upos="NOUN",
                features={"Gender": "Fem", "Number": "Sing"},
            ),
            Token(
                text="x",
                lemma="x",
                upos="ADJ",
                features={"Gender": "Com", "Number": "Sing"},
            ),
            Token(
                text="hus",
                lemma="hus",
                upos="NOUN",
                features={"Gender": "Neut", "Number": "Sing"},
            ),
        ),
    )
    comparison_tokens = (
        (
            gold_tokens[0][0],
            Token(
                text="boka",
                lemma="bok",
                upos="NOUN",
                features={"Gender": "Masc", "Number": "Sing"},
            ),
            Token(
                text="x",
                lemma="x",
                upos="ADJ",
                features={"Gender": "Com", "Number": "Plur"},
            ),
            gold_tokens[0][3],
        ),
    )
    accumulator = MorphologyErrorAuditAccumulator(
        schema=schema,
        feature_name="Gender",
        reference_batches=(build_universal_dependencies_reference_batch(gold_tokens),),
        comparison_reference_batches=(
            build_universal_dependencies_reference_batch(comparison_tokens),
        ),
        frequency_profile=TokenFrequencyProfile.from_token_sequences(
            (("den",) * 6 + ("boka", "hus"),)
        ),
    )

    def categorical_predictions(
        feature_name: str,
        labels: tuple[str, ...],
    ) -> torch.Tensor:
        feature = next(
            feature
            for feature in schema.morphology.features
            if feature.name == feature_name
        )
        return torch.tensor(
            [
                [
                    [label == selected_label for label in feature.labels]
                    for selected_label in labels
                ]
            ],
            dtype=torch.bool,
        )

    accumulator.add(
        predictions=TokenTaskPredictionBatch(
            upos_ids=torch.tensor(
                [
                    [
                        schema.upos.label_id_for("DET"),
                        schema.upos.label_id_for("NOUN"),
                        schema.upos.label_id_for("ADJ"),
                        schema.upos.label_id_for("NOUN"),
                    ]
                ],
                dtype=torch.long,
            ),
            morphology_predictions=(
                categorical_predictions(
                    "Gender",
                    ("Fem", "Masc", NO_MORPHOLOGY_VALUE, "Neut"),
                ),
                categorical_predictions(
                    "Number",
                    ("Sing", "Sing", "Sing", "Sing"),
                ),
            ),
            lemma_rule_ids=torch.zeros((1, 4), dtype=torch.long),
            token_mask=torch.ones((1, 4), dtype=torch.bool),
        )
    )

    audit = accumulator.finish()

    assert audit.feature_name == "Gender"
    assert audit.token_count == 4
    assert audit.error_count == 3
    assert audit.comparison_feature_correct_count == 2
    assert audit.comparison_bundle_correct_count == 1
    assert tuple(
        (count.name, count.count) for count in audit.frequency_class_counts
    ) == (("frequent", 1), ("oov", 1), ("rare", 1))
    assert tuple((count.name, count.count) for count in audit.gold_upos_counts) == (
        ("ADJ", 1),
        ("DET", 1),
        ("NOUN", 1),
    )
    assert audit.errors[0].previous_gold_upos == "<BOS>"
    assert audit.errors[2].next_gold_upos == "NOUN"


def test_morphology_error_audit_rejects_unknown_feature() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(("NOUN",)),
        morphology=build_morphology_schema(({"Gender": "Neut"},)),
        lemma_rules=build_lemma_rule_schema((("hus", "hus"),)),
    )

    try:
        MorphologyErrorAuditAccumulator(
            schema=schema,
            feature_name="Unknown",
            reference_batches=(
                build_universal_dependencies_reference_batch(
                    (
                        (
                            Token(
                                text="hus",
                                lemma="hus",
                                upos="NOUN",
                                features={"Gender": "Neut"},
                            ),
                        ),
                    )
                ),
            ),
            frequency_profile=TokenFrequencyProfile.from_token_sequences((("hus",),)),
        )
    except ValueError as error:
        assert str(error) == "Unknown morphology audit feature: 'Unknown'"
    else:
        raise AssertionError("Unknown audit feature must be rejected.")
