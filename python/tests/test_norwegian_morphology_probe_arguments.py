from pathlib import Path

from prism.languages.norwegian.probe_morphology_heads import (
    parse_probe_arguments,
)
from prism.training import MorphologyHeadProbeArchitecture


def test_parse_probe_arguments_selects_reusable_diagnostic_policy() -> None:
    arguments = parse_probe_arguments(
        (
            "--checkpoint",
            "runs/no-student/best.pt",
            "--analysis",
            "runs/no-student/probe.json",
            "--representation-cache",
            "runs/no-student/probe-cache.pt",
            "--evaluation-language-tag",
            "nb",
            "--evaluation-language-tag",
            "nn",
            "--architecture",
            "linear",
            "--architecture",
            "feature-mlp",
            "--epoch-count",
            "6",
            "--random-seed",
            "17",
        )
    )

    assert arguments.checkpoint_path == Path("runs/no-student/best.pt")
    assert arguments.analysis_path == Path("runs/no-student/probe.json")
    assert arguments.representation_cache_path == Path("runs/no-student/probe-cache.pt")
    assert arguments.evaluation_language_tags == ("nb", "nn")
    assert arguments.architectures == (
        MorphologyHeadProbeArchitecture.LINEAR,
        MorphologyHeadProbeArchitecture.FEATURE_MLP,
    )
    assert arguments.config.epoch_count == 6
    assert arguments.config.random_seed == 17
    assert arguments.morphology_logit_correction_strength == 1.0
