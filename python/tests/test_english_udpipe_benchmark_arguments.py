from pathlib import Path

from prism.languages.english.benchmark_udpipe import (
    UDPIPE_2_17_MODELS,
    parse_udpipe_benchmark_arguments,
)


def test_english_udpipe_model_inventory() -> None:
    assert UDPIPE_2_17_MODELS == {"en": "english-ewt-ud-2.17-251125"}


def test_english_udpipe_defaults_to_2_17_dev_split() -> None:
    arguments = parse_udpipe_benchmark_arguments(["--treebank-release", "2.17"])
    assert arguments.language_tag == "en"
    assert arguments.model == "english-ewt-ud-2.17-251125"
    assert arguments.treebank_release == "2.17"
    assert arguments.gold_path == Path(
        "data/raw/ud-2.17/UD_English-EWT/en_ewt-ud-dev.conllu"
    )


def test_english_udpipe_prediction_and_analysis_paths() -> None:
    arguments = parse_udpipe_benchmark_arguments(["--treebank-release", "2.17"])
    run_directory = Path("runs/udpipe-2.17-251125/ud-2.17")
    assert arguments.prediction_path == run_directory / "en-development.conllu"
    assert arguments.analysis_path == run_directory / "en-development-analysis.json"
