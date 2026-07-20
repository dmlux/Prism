from pathlib import Path

from prism.languages.norwegian.train_baseline import (
    parse_training_arguments,
)


def test_parse_training_arguments_preserves_baseline_variants() -> None:
    default_arguments = parse_training_arguments(())

    assert default_arguments.language_tag == "nb"
    assert default_arguments.checkpoint_path == Path("runs/nb-student-baseline/best.pt")
    assert default_arguments.morphology_positive_weight_cap is None

    weighted_arguments = parse_training_arguments(
        (
            "--checkpoint",
            "runs/nb-student-weighted/best.pt",
            "--morphology-positive-weight-cap",
            "10.0",
        )
    )

    assert weighted_arguments.checkpoint_path == Path(
        "runs/nb-student-weighted/best.pt"
    )
    assert weighted_arguments.morphology_positive_weight_cap == 10.0

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
