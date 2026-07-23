from pathlib import Path

from prism.languages.norwegian.evaluate_baseline import (
    parse_evaluation_arguments,
)


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
            "--disable-morphology-agreement-refiner",
            "--morphology-error-audit-feature",
            "Gender",
            "--morphology-error-audit-comparison",
            "runs/udpipe/nb-development.conllu",
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
    assert arguments.disable_morphology_agreement_refiner
    assert arguments.morphology_error_audit_feature == "Gender"
    assert arguments.morphology_error_audit_comparison_path == Path(
        "runs/udpipe/nb-development.conllu"
    )
