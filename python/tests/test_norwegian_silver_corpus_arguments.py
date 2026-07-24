from pathlib import Path

import pytest

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

    assert arguments.source == "nbdigital-nob"
    assert arguments.archive_path == Path("data/raw/nbdigital/corpus.tar.gz")
    assert arguments.output_path == Path(
        "data/processed/nbdigital-nob-free/pretokenized.jsonl"
    )
    assert arguments.minimum_ocr_confidence == 0.97
    assert arguments.maximum_token_count == 96


def test_parse_sakspapir_preparation_arguments() -> None:
    arguments = parse_preparation_arguments(
        (
            "--source",
            "sakspapir-nno",
            "--archive",
            "data/raw/sakspapir_nno_01.tar.gz",
        )
    )

    assert arguments.source == "sakspapir-nno"
    assert arguments.output_path == Path(
        "data/processed/sakspapir-nno/pretokenized.jsonl"
    )
    assert arguments.manifest_path == Path(
        "data/processed/sakspapir-nno/manifest.json"
    )
    assert arguments.minimum_ocr_confidence is None
    assert arguments.maximum_token_count == 128


def test_sakspapir_rejects_ocr_confidence_option() -> None:
    with pytest.raises(SystemExit):
        parse_preparation_arguments(
            (
                "--source",
                "sakspapir-nno",
                "--archive",
                "data/raw/sakspapir_nno_01.tar.gz",
                "--minimum-ocr-confidence",
                "0.9",
            )
        )
