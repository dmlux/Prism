from pathlib import Path

from prism.languages.norwegian.calibrate_baseline import (
    parse_calibration_arguments,
)


def test_parse_calibration_arguments_defaults() -> None:
    arguments = parse_calibration_arguments(
        ("--checkpoint", "runs/example/best.pt")
    )

    assert arguments.checkpoint_path == Path("runs/example/best.pt")
    assert arguments.calibration_path is None
    assert arguments.language_tag == "no"
    assert arguments.device == "mps"
    assert arguments.treebank_release == "current"
    assert arguments.morphology_logit_correction_strength == 0.0


def test_parse_calibration_arguments_accepts_explicit_values() -> None:
    arguments = parse_calibration_arguments(
        (
            "--checkpoint",
            "runs/example/best.pt",
            "--calibration",
            "runs/example/calibration-corrected.json",
            "--language-tag",
            "nb",
            "--device",
            "cpu",
            "--morphology-logit-correction-strength",
            "1.0",
        )
    )

    assert arguments.calibration_path == Path(
        "runs/example/calibration-corrected.json"
    )
    assert arguments.language_tag == "nb"
    assert arguments.device == "cpu"
    assert arguments.morphology_logit_correction_strength == 1.0
