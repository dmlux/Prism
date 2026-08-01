from pathlib import Path

from prism.languages.norwegian.evaluate_baseline import (
    parse_evaluation_arguments,
)


def test_parse_evaluation_arguments_resolves_split() -> None:
    default_arguments = parse_evaluation_arguments(())
    test_arguments = parse_evaluation_arguments(("--split", "test"))

    assert default_arguments.evaluation_split == "development"
    assert test_arguments.evaluation_split == "test"


def test_parse_evaluation_arguments_accepts_language_profile() -> None:
    arguments = parse_evaluation_arguments(
        (
            "--language-tag",
            "nn",
            "--checkpoint",
            "runs/nn-student-weighted/best.pt",
            "--analysis",
            "runs/nn-student-weighted/development-analysis.json",
            "--device",
            "cpu",
            "--treebank-release",
            "2.17",
            "--morphology-logit-correction-strength",
            "0.5",
            "--ud-morphology-policy",
            "treebank",
            "--disable-morphology-bundle-reranker",
            "--morphology-error-audit-feature",
            "Gender",
            "--morphology-error-audit-comparison",
            "runs/udpipe/nb-development.conllu",
            "--morphology-feature-comparison",
            "runs/udpipe/nn-development.conllu",
            "--morphology-feature-comparison-name",
            "UDPipe 2.17",
        )
    )

    assert arguments.language_tag == "nn"
    assert arguments.checkpoint_path == Path("runs/nn-student-weighted/best.pt")
    assert arguments.analysis_path == Path(
        "runs/nn-student-weighted/development-analysis.json"
    )
    assert arguments.device == "cpu"
    assert arguments.treebank_release == "2.17"
    assert arguments.morphology_logit_correction_strength == 0.5
    assert arguments.ud_morphology_policy == "treebank"
    assert arguments.disable_morphology_bundle_reranker
    assert arguments.morphology_error_audit_feature == "Gender"
    assert arguments.morphology_error_audit_comparison_path == Path(
        "runs/udpipe/nb-development.conllu"
    )
    assert arguments.morphology_feature_comparison_path == Path(
        "runs/udpipe/nn-development.conllu"
    )
    assert arguments.morphology_feature_comparison_name == "UDPipe 2.17"
    assert not arguments.task_interaction_audit
    assert arguments.task_interaction_gradient_batch_count == 16


def test_parse_evaluation_arguments_accepts_task_interaction_audit() -> None:
    arguments = parse_evaluation_arguments(
        (
            "--task-interaction-audit",
            "--task-interaction-gradient-batches",
            "12",
        )
    )

    assert arguments.task_interaction_audit
    assert arguments.task_interaction_gradient_batch_count == 12
