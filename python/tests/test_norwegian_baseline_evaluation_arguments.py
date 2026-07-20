from pathlib import Path

from prism.languages.norwegian.evaluate_baseline import (
    parse_evaluation_arguments,
)


def test_parse_evaluation_arguments_accepts_weighted_checkpoint() -> None:
    arguments = parse_evaluation_arguments(
        (
            "--checkpoint",
            "runs/nb-student-weighted/best.pt",
            "--analysis",
            "runs/nb-student-weighted/development-analysis.json",
        )
    )

    assert arguments.checkpoint_path == Path("runs/nb-student-weighted/best.pt")
    assert arguments.analysis_path == Path(
        "runs/nb-student-weighted/development-analysis.json"
    )
