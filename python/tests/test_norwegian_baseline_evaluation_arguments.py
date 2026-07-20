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
        )
    )

    assert arguments.language_tag == "nn"
    assert arguments.checkpoint_path == Path("runs/nn-student-weighted/best.pt")
    assert arguments.analysis_path == Path(
        "runs/nn-student-weighted/development-analysis.json"
    )
