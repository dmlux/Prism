from pathlib import Path

from prism.languages.norwegian.prepare_silver_corpus import (
    parse_preparation_arguments,
)


def test_parse_silver_corpus_preparation_arguments() -> None:
    arguments = parse_preparation_arguments(
        (
            "--archive",
            "data/raw/nbdigital/corpus.tar.gz",
            "--minimum-ocr-confidence",
            "0.97",
            "--maximum-token-count",
            "96",
        )
    )

    assert arguments.archive_path == Path("data/raw/nbdigital/corpus.tar.gz")
    assert arguments.output_path == Path(
        "data/processed/nbdigital-nob-free/pretokenized.jsonl"
    )
    assert arguments.minimum_ocr_confidence == 0.97
    assert arguments.maximum_token_count == 96
