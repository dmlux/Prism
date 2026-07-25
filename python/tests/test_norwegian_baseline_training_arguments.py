from pathlib import Path

import pytest

from prism.modeling import (
    BackboneLayerAggregationStrategy,
    MorphologyBundleLossGradientScope,
    MorphologyBundleScorerArchitecture,
    MorphologyPreHeadArchitecture,
    TokenPoolingStrategy,
    TokenTaskHeadArchitecture,
)
from prism.training import CheckpointSelectionMetric

from prism.languages.norwegian.train_baseline import (
    parse_training_arguments,
)
from prism.training import TokenTaskDistillationPolicy


def test_parse_training_arguments_preserves_baseline_variants() -> None:
    default_arguments = parse_training_arguments(())

    assert default_arguments.language_tag == "nb"
    assert default_arguments.checkpoint_path == Path("runs/nb-student-baseline/best.pt")
    assert default_arguments.morphology_weight_cap is None
    assert default_arguments.distillation_policy == (
        TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        )
    )
    assert default_arguments.token_pooling_strategy is TokenPoolingStrategy.MEAN
    assert (
        default_arguments.token_task_head_architecture
        is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
    )
    assert (
        default_arguments.morphology_pre_head_architecture
        is MorphologyPreHeadArchitecture.SHARED_MLP
    )
    assert default_arguments.epoch_count == 12
    assert default_arguments.treebank_release == "current"
    assert default_arguments.morphology_bundle_candidate_count == 0
    assert (
        default_arguments.morphology_bundle_scorer_architecture
        is MorphologyBundleScorerArchitecture.LINEAR
    )
    assert default_arguments.morphology_bundle_loss_weight == 0.0
    assert (
        default_arguments.morphology_bundle_loss_gradient_scope
        is MorphologyBundleLossGradientScope.FULL
    )
    assert default_arguments.early_stopping_patience == 4
    assert (
        default_arguments.backbone_layer_aggregation
        is BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR
    )

    weighted_arguments = parse_training_arguments(
        (
            "--checkpoint",
            "runs/nb-student-weighted/best.pt",
            "--morphology-weight-cap",
            "10.0",
        )
    )

    assert weighted_arguments.checkpoint_path == Path(
        "runs/nb-student-weighted/best.pt"
    )
    assert weighted_arguments.morphology_weight_cap == 10.0

    ud_2_17_arguments = parse_training_arguments(("--treebank-release", "2.17"))
    assert ud_2_17_arguments.treebank_release == "2.17"

    first_pooling_arguments = parse_training_arguments(
        (
            "--token-pooling",
            "first",
        )
    )

    assert first_pooling_arguments.token_pooling_strategy is TokenPoolingStrategy.FIRST

    linear_arguments = parse_training_arguments(
        (
            "--task-head-architecture",
            "linear",
            "--epoch-count",
            "8",
        )
    )

    assert (
        linear_arguments.token_task_head_architecture
        is TokenTaskHeadArchitecture.LINEAR
    )
    assert linear_arguments.epoch_count == 8

    character_arguments = parse_training_arguments(
        (
            "--task-head-architecture",
            "wide-shared-mlp-structured-morphology-character-cnn",
        )
    )

    assert (
        character_arguments.token_task_head_architecture
        is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
    )

    morphology_pre_head_arguments = parse_training_arguments(
        (
            "--morphology-pre-head-architecture",
            "shared-mlp",
        )
    )
    assert (
        morphology_pre_head_arguments.morphology_pre_head_architecture
        is MorphologyPreHeadArchitecture.SHARED_MLP
    )

    identity_morphology_pre_head_arguments = parse_training_arguments(
        (
            "--morphology-pre-head-architecture",
            "identity",
        )
    )
    assert (
        identity_morphology_pre_head_arguments.morphology_pre_head_architecture
        is MorphologyPreHeadArchitecture.IDENTITY
    )

    reranker_arguments = parse_training_arguments(
        ("--morphology-bundle-candidate-count", "32"),
    )
    assert reranker_arguments.morphology_bundle_candidate_count == 32
    assert (
        reranker_arguments.morphology_bundle_scorer_architecture
        is MorphologyBundleScorerArchitecture.LINEAR
    )
    assert reranker_arguments.morphology_bundle_loss_weight == 0.0
    assert (
        reranker_arguments.morphology_bundle_loss_gradient_scope
        is MorphologyBundleLossGradientScope.FULL
    )

    compositional_reranker_arguments = parse_training_arguments(
        (
            "--morphology-bundle-candidate-count",
            "32",
            "--morphology-bundle-scorer-architecture",
            "compositional-mlp",
        ),
    )
    assert (
        compositional_reranker_arguments.morphology_bundle_scorer_architecture
        is MorphologyBundleScorerArchitecture.COMPOSITIONAL_MLP
    )

    with pytest.raises(SystemExit):
        parse_training_arguments(
            (
                "--morphology-bundle-scorer-architecture",
                "compositional-mlp",
            ),
        )

    bundle_loss_arguments = parse_training_arguments(
        (
            "--morphology-bundle-candidate-count",
            "32",
            "--morphology-bundle-loss-weight",
            "0.1",
            "--early-stopping-patience",
            "0",
        ),
    )
    assert bundle_loss_arguments.morphology_bundle_loss_weight == 0.1
    assert (
        bundle_loss_arguments.morphology_bundle_loss_gradient_scope
        is MorphologyBundleLossGradientScope.MORPHOLOGY
    )
    assert bundle_loss_arguments.early_stopping_patience is None

    full_gradient_arguments = parse_training_arguments(
        (
            "--morphology-bundle-candidate-count",
            "32",
            "--morphology-bundle-loss-weight",
            "0.1",
            "--morphology-bundle-loss-gradient-scope",
            "full",
        ),
    )
    assert (
        full_gradient_arguments.morphology_bundle_loss_gradient_scope
        is MorphologyBundleLossGradientScope.FULL
    )

    with pytest.raises(SystemExit):
        parse_training_arguments(
            (
                "--morphology-bundle-candidate-count",
                "32",
                "--morphology-bundle-loss-gradient-scope",
                "morphology",
            ),
        )

    layer_mix_arguments = parse_training_arguments(
        (
            "--backbone-layer-aggregation",
            "learned-last-four",
        )
    )

    assert (
        layer_mix_arguments.backbone_layer_aggregation
        is BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR
    )

    final_layer_arguments = parse_training_arguments(
        (
            "--backbone-layer-aggregation",
            "last",
        )
    )

    assert (
        final_layer_arguments.backbone_layer_aggregation
        is BackboneLayerAggregationStrategy.LAST
    )

    nynorsk_arguments = parse_training_arguments(
        (
            "--language-tag",
            "nn",
            "--checkpoint",
            "runs/nn-student-baseline/best.pt",
        )
    )

    assert nynorsk_arguments.language_tag == "nn"
    assert nynorsk_arguments.checkpoint_path == Path("runs/nn-student-baseline/best.pt")

    joint_arguments = parse_training_arguments(
        (
            "--language-tag",
            "no",
            "--checkpoint",
            "runs/no-student-weighted/best.pt",
        )
    )

    assert joint_arguments.language_tag == "no"
    assert joint_arguments.checkpoint_path == Path("runs/no-student-weighted/best.pt")


def test_parse_training_arguments_accepts_teacher_role() -> None:
    arguments = parse_training_arguments(
        (
            "--language-tag",
            "no",
            "--model-role",
            "teacher",
            "--checkpoint",
            "runs/no-teacher-base/best.pt",
        )
    )

    assert arguments.language_tag == "no"
    assert arguments.model_role == "teacher"
    assert arguments.checkpoint_path == Path("runs/no-teacher-base/best.pt")


def test_parse_training_arguments_accepts_distillation_policy() -> None:
    arguments = parse_training_arguments(
        (
            "--language-tag",
            "no",
            "--model-role",
            "student",
            "--checkpoint",
            "runs/no-student-distilled/best.pt",
            "--teacher-checkpoint",
            "runs/no-teacher-base/best.pt",
            "--distillation-temperature",
            "2.0",
            "--distillation-weight",
            "0.5",
        )
    )

    assert arguments.teacher_checkpoint_path == Path("runs/no-teacher-base/best.pt")
    assert arguments.distillation_policy == TokenTaskDistillationPolicy.uniform(
        temperature=2.0,
        weight=0.5,
    )


def test_parse_training_arguments_accepts_task_specific_distillation_policy() -> None:
    arguments = parse_training_arguments(
        (
            "--upos-distillation-temperature",
            "1.0",
            "--morphology-distillation-temperature",
            "1.5",
            "--lemma-rule-distillation-temperature",
            "2.0",
            "--upos-distillation-weight",
            "0.05",
            "--morphology-distillation-weight",
            "0.2",
            "--lemma-rule-distillation-weight",
            "0.1",
        )
    )

    assert arguments.distillation_policy == TokenTaskDistillationPolicy(
        upos_temperature=1.0,
        morphology_temperature=1.5,
        lemma_rule_temperature=2.0,
        upos_weight=0.05,
        morphology_weight=0.2,
        lemma_rule_weight=0.1,
    )


def test_parse_training_arguments_accepts_decoupled_distillation() -> None:
    arguments = parse_training_arguments(
        (
            "--categorical-distillation-objective",
            "dkd",
            "--dkd-target-class-weight",
            "1.0",
            "--dkd-non-target-class-weight",
            "2.0",
        )
    )

    assert arguments.distillation_policy == TokenTaskDistillationPolicy.uniform(
        temperature=1.0,
        weight=0.1,
        categorical_objective="dkd",
        target_class_weight=1.0,
        non_target_class_weight=2.0,
    )


def test_parse_training_arguments_rejects_non_positive_epoch_count() -> None:
    with pytest.raises(SystemExit):
        parse_training_arguments(("--epoch-count", "0"))


def test_parse_training_arguments_resolves_checkpoint_selection_metric() -> None:
    default_arguments = parse_training_arguments(())
    discrete_arguments = parse_training_arguments(
        ("--checkpoint-selection-metric", "development-task-accuracy")
    )

    assert (
        default_arguments.checkpoint_selection_metric
        is CheckpointSelectionMetric.DEVELOPMENT_LOSS
    )
    assert (
        discrete_arguments.checkpoint_selection_metric
        is CheckpointSelectionMetric.DEVELOPMENT_TASK_ACCURACY
    )

    with pytest.raises(SystemExit):
        parse_training_arguments(("--checkpoint-selection-metric", "gender-only"))


def test_parse_training_arguments_resolves_secondary_selection_metric() -> None:
    default_arguments = parse_training_arguments(())
    dual_arguments = parse_training_arguments(
        (
            "--secondary-checkpoint-selection-metric",
            "development-task-accuracy",
        )
    )

    assert default_arguments.secondary_checkpoint_selection_metric is None
    assert (
        dual_arguments.secondary_checkpoint_selection_metric
        is CheckpointSelectionMetric.DEVELOPMENT_TASK_ACCURACY
    )

    with pytest.raises(SystemExit):
        parse_training_arguments(
            (
                "--checkpoint-selection-metric",
                "development-task-accuracy",
                "--secondary-checkpoint-selection-metric",
                "development-task-accuracy",
            )
        )
