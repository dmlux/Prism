from pathlib import Path

import pytest

from prism.languages.norwegian.benchmark_speed import (
    parse_speed_benchmark_arguments,
)


def test_parse_speed_benchmark_arguments_resolves_defaults() -> None:
    arguments = parse_speed_benchmark_arguments(
        ("--checkpoint", "runs/no-student/best.pt")
    )

    assert arguments.checkpoint_path == Path("runs/no-student/best.pt")
    assert arguments.language_tag == "nb"
    assert arguments.evaluation_split == "test"
    assert arguments.device == "cpu"
    assert arguments.batch_sizes == (1, 32)
    assert arguments.warmup_batch_count == 8
    assert arguments.morphology_logit_correction_strength == 0.25
    assert arguments.analysis_path == Path("runs/no-student/nb-test-speed-cpu.json")


def test_parse_speed_benchmark_arguments_accepts_configuration() -> None:
    arguments = parse_speed_benchmark_arguments(
        (
            "--checkpoint",
            "runs/no-student/best.pt",
            "--language-tag",
            "nn",
            "--split",
            "development",
            "--device",
            "mps",
            "--batch-size",
            "1",
            "--batch-size",
            "64",
            "--warmup-batches",
            "2",
            "--analysis",
            "runs/no-student/speed.json",
        )
    )

    assert arguments.language_tag == "nn"
    assert arguments.evaluation_split == "development"
    assert arguments.device == "mps"
    assert arguments.batch_sizes == (1, 64)
    assert arguments.warmup_batch_count == 2
    assert arguments.analysis_path == Path("runs/no-student/speed.json")


def test_parse_speed_benchmark_arguments_rejects_invalid_values() -> None:
    with pytest.raises(SystemExit):
        parse_speed_benchmark_arguments(
            ("--checkpoint", "runs/no-student/best.pt", "--batch-size", "0")
        )
    with pytest.raises(SystemExit):
        parse_speed_benchmark_arguments(
            ("--checkpoint", "runs/no-student/best.pt", "--warmup-batches", "-1")
        )
    with pytest.raises(SystemExit):
        parse_speed_benchmark_arguments(())
