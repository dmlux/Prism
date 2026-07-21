from pathlib import Path

import pytest

from prism.modeling import TokenPoolingStrategy, TokenTaskHeadArchitecture
from prism.languages.norwegian.train_baseline import (
    parse_training_arguments,
)


def test_parse_training_arguments_preserves_baseline_variants() -> None:
    default_arguments = parse_training_arguments(())

    assert default_arguments.language_tag == "nb"
    assert default_arguments.checkpoint_path == Path("runs/nb-student-baseline/best.pt")
    assert default_arguments.morphology_weight_cap is None
    assert default_arguments.token_pooling_strategy is TokenPoolingStrategy.MEAN
    assert (
        default_arguments.token_task_head_architecture
        is TokenTaskHeadArchitecture.SHARED_MLP
    )
    assert default_arguments.epoch_count == 5

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
    assert arguments.distillation_temperature == 2.0
    assert arguments.distillation_weight == 0.5


def test_parse_training_arguments_rejects_non_positive_epoch_count() -> None:
    with pytest.raises(SystemExit):
        parse_training_arguments(("--epoch-count", "0"))
