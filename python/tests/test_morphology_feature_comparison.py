import torch

from prism.conllu import Token
from prism.evaluation import (
    MorphologyFeatureComparisonAccumulator,
    TokenFrequencyProfile,
    build_universal_dependencies_reference_batch,
    serialize_morphology_feature_comparison_report,
)
from prism.modeling.outputs import TokenTaskPredictionBatch
from prism.schema import (
    NO_MORPHOLOGY_VALUE,
    TokenTaskSchema,
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
)


def test_morphology_feature_comparison_reports_both_systems_and_slices() -> None:
    schema = TokenTaskSchema(
        upos=build_upos_schema(("CCONJ", "DET", "NOUN")),
        morphology=build_morphology_schema(
            (
                {"Gender": "Masc", "Number": "Sing"},
                {"Gender": "Fem", "Number": "Sing"},
                {"Gender": "Neut", "Number": "Sing"},
                {},
            )
        ),
        lemma_rules=build_lemma_rule_schema(
            (("den", "den"), ("boka", "bok"), ("hus", "hus"), ("og", "og"))
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
                text="hus",
                lemma="hus",
                upos="NOUN",
                features={"Gender": "Neut", "Number": "Sing"},
            ),
            Token(text="og", lemma="og", upos="CCONJ", features={}),
        ),
    )
    comparison_tokens = (
        (
            Token(
                text="den",
                lemma="den",
                upos="DET",
                features={"Gender": "Fem", "Number": "Sing"},
            ),
            gold_tokens[0][1],
            Token(
                text="hus",
                lemma="hus",
                upos="NOUN",
                features={"Gender": "Neut"},
            ),
            gold_tokens[0][3],
        ),
    )
    accumulator = MorphologyFeatureComparisonAccumulator(
        schema=schema,
        reference_batches=(build_universal_dependencies_reference_batch(gold_tokens),),
        comparison_reference_batches=(
            build_universal_dependencies_reference_batch(comparison_tokens),
        ),
        frequency_profile=TokenFrequencyProfile.from_token_sequences(
            (("den",) * 6 + ("boka",),)
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
                        schema.upos.label_id_for("NOUN"),
                        schema.upos.label_id_for("CCONJ"),
                    ]
                ],
                dtype=torch.long,
            ),
            morphology_predictions=(
                categorical_predictions(
                    "Gender",
                    ("Masc", "Masc", "Neut", NO_MORPHOLOGY_VALUE),
                ),
                categorical_predictions(
                    "Number",
                    ("Sing", "Sing", NO_MORPHOLOGY_VALUE, NO_MORPHOLOGY_VALUE),
                ),
            ),
            lemma_rule_ids=torch.zeros((1, 4), dtype=torch.long),
            token_mask=torch.ones((1, 4), dtype=torch.bool),
        )
    )

    report = accumulator.finish()

    assert report.token_count == 4
    gender, number = report.features
    assert gender.feature_name == "Gender"
    assert gender.model.overall_accuracy == 0.75
    assert gender.comparison.overall_accuracy == 0.75
    assert gender.model.annotated_accuracy == 2 / 3
    assert gender.comparison.annotated_accuracy == 2 / 3
    assert gender.model.wrong_bundle_count == 2
    assert gender.model.feature_error_in_wrong_bundle_count == 1
    assert gender.model.wrong_bundle_error_share == 0.5
    assert number.feature_name == "Number"
    assert number.model.overall_accuracy == 0.75

    rare_gender = gender.frequency_slices[0]
    oov_gender = gender.frequency_slices[1]
    assert rare_gender.name == "rare"
    assert rare_gender.model.overall_accuracy == 0.0
    assert rare_gender.comparison.overall_accuracy == 1.0
    assert oov_gender.name == "oov"
    assert oov_gender.model.overall_accuracy == 1.0
    assert oov_gender.comparison.overall_accuracy == 1.0

    serialized = serialize_morphology_feature_comparison_report(report)
    serialized_gender = serialized["features"][0]
    assert serialized_gender["model"]["overall_accuracy"] == 0.75
    assert serialized_gender["model"]["values"][0]["value"] == "Fem"
