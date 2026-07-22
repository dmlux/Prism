from pathlib import Path

from prism.languages.norwegian.benchmark_udpipe import (
    parse_udpipe_benchmark_arguments,
)


def test_udpipe_benchmark_arguments_select_versioned_nynorsk_model() -> None:
    arguments = parse_udpipe_benchmark_arguments(
        (
            "--language-tag",
            "nn",
            "--treebank-release",
            "2.17",
            "--prediction",
            "runs/udpipe/nn.conllu",
            "--analysis",
            "runs/udpipe/nn.json",
            "--reuse-prediction",
        )
    )

    assert arguments.language_tag == "nn"
    assert arguments.model == "norwegian-nynorsk-ud-2.17-251125"
    assert arguments.prediction_path == Path("runs/udpipe/nn.conllu")
    assert arguments.analysis_path == Path("runs/udpipe/nn.json")
    assert arguments.reuse_prediction
    assert arguments.treebank_release == "2.17"
