from pathlib import Path

from prism.languages.norwegian.export_artifact import (
    parse_artifact_export_arguments,
)


def test_parse_artifact_export_arguments_defaults() -> None:
    arguments = parse_artifact_export_arguments(
        (
            "--checkpoint",
            "runs/example/best.pt",
            "--artifact-version",
            "0.1.0",
        )
    )

    assert arguments.checkpoint_path == Path("runs/example/best.pt")
    assert arguments.artifact_version == "0.1.0"
    assert arguments.output_root == Path("models")
    assert arguments.language_tag == "no"
    assert arguments.treebank_release == "current"
    assert arguments.morphology_logit_correction_strength == 0.25
    assert arguments.batch_size == 8
    assert arguments.subword_count == 160
    assert arguments.token_count == 96
    assert arguments.fixture_sentence_count == 8
    assert arguments.fixture_lemma_top_k == 8
    assert arguments.parity_tolerance == 5e-3
    assert arguments.overwrite is False


def test_parse_artifact_export_arguments_accepts_explicit_values() -> None:
    arguments = parse_artifact_export_arguments(
        (
            "--checkpoint",
            "runs/example/best.pt",
            "--artifact-version",
            "0.2.0",
            "--output-root",
            "build/artifacts",
            "--language-tag",
            "nb",
            "--treebank-release",
            "2.17",
            "--morphology-logit-correction-strength",
            "0.0",
            "--batch-size",
            "4",
            "--subword-count",
            "128",
            "--token-count",
            "64",
            "--fixture-sentence-count",
            "2",
            "--parity-tolerance",
            "5e-4",
            "--overwrite",
        )
    )

    assert arguments.artifact_version == "0.2.0"
    assert arguments.output_root == Path("build/artifacts")
    assert arguments.language_tag == "nb"
    assert arguments.treebank_release == "2.17"
    assert arguments.morphology_logit_correction_strength == 0.0
    assert arguments.batch_size == 4
    assert arguments.subword_count == 128
    assert arguments.token_count == 64
    assert arguments.fixture_sentence_count == 2
    assert arguments.parity_tolerance == 5e-4
    assert arguments.overwrite is True
