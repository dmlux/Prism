import io
import json
import tarfile
from pathlib import Path

import pytest

from prism.data import (
    SAKSPAPIR_CORPUS_ID,
    SAKSPAPIR_LICENSE_ID,
    SAKSPAPIR_LICENSE_URL,
    SAKSPAPIR_SOURCE_URL,
    iter_sakspapir_silver_sentences,
    load_silver_corpus_manifest,
    sentence_fingerprint,
    sha256_file,
    validate_silver_corpus,
    write_pretokenized_silver_corpus,
)
from prism.data.sakspapir import _iter_json_object_items
from prism.data.segmentation import SentenceExtractionPolicy


_POLICY = SentenceExtractionPolicy(
    abbreviation_tokens=frozenset({"f.eks."}),
    minimum_token_count=3,
    maximum_token_count=32,
    minimum_letter_token_ratio=0.5,
)


def _write_archive(path: Path) -> None:
    corpus = {
        "urn:uuid:0001": [
            ["2", "nno", "Denne setninga står på side to.\n"],
            ["1", "nno", "MØTEINNKALLING\nFyrste sida har vanleg prosa.\n"],
            ["3", "nob", "Denne bokmålssiden skal ignoreres helt.\n"],
        ],
        "urn:uuid:0002": [
            ["1", "nno", "Fyrste sida har vanleg prosa.\nEit anna innhald òg.\n"],
        ],
    }
    payload = json.dumps(corpus, ensure_ascii=False).encode("utf-8")
    with tarfile.open(path, mode="w:gz") as archive:
        member = tarfile.TarInfo("sakspapir_nno_01.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
        extra = "urn:uuid:0001\thttps://example.no/dokument.pdf\n".encode()
        url_member = tarfile.TarInfo("urn_url.txt")
        url_member.size = len(extra)
        archive.addfile(url_member, io.BytesIO(extra))


def test_sakspapir_filters_language_orders_pages_and_deduplicates(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "sakspapir.tar.gz"
    _write_archive(archive_path)

    sentences = tuple(
        iter_sakspapir_silver_sentences(
            archive_path=archive_path,
            extraction_policy=_POLICY,
        )
    )

    assert tuple(
        (sentence.document_id, sentence.sentence_index, sentence.model_input.tokens)
        for sentence in sentences
    ) == (
        (
            "urn:uuid:0001",
            0,
            ("Fyrste", "sida", "har", "vanleg", "prosa", "."),
        ),
        (
            "urn:uuid:0001",
            1,
            ("Denne", "setninga", "står", "på", "side", "to", "."),
        ),
        (
            "urn:uuid:0002",
            0,
            ("Eit", "anna", "innhald", "òg", "."),
        ),
    )


def test_sakspapir_respects_excluded_fingerprints(tmp_path: Path) -> None:
    archive_path = tmp_path / "sakspapir.tar.gz"
    _write_archive(archive_path)
    all_sentences = tuple(
        iter_sakspapir_silver_sentences(
            archive_path=archive_path,
            extraction_policy=_POLICY,
        )
    )

    filtered = tuple(
        iter_sakspapir_silver_sentences(
            archive_path=archive_path,
            extraction_policy=_POLICY,
            excluded_sentence_fingerprints=(
                sentence_fingerprint(all_sentences[0].model_input),
            ),
        )
    )

    assert tuple(sentence.model_input.tokens for sentence in filtered) == (
        ("Denne", "setninga", "står", "på", "side", "to", "."),
        ("Eit", "anna", "innhald", "òg", "."),
    )


def test_sakspapir_manifest_round_trip(tmp_path: Path) -> None:
    archive_path = tmp_path / "sakspapir.tar.gz"
    _write_archive(archive_path)
    output_path = tmp_path / "sentences.jsonl"
    manifest_path = tmp_path / "manifest.json"

    written_manifest = write_pretokenized_silver_corpus(
        sentences=iter_sakspapir_silver_sentences(
            archive_path=archive_path,
            extraction_policy=_POLICY,
        ),
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_id=SAKSPAPIR_CORPUS_ID,
        language_tag="nn",
        source_url=SAKSPAPIR_SOURCE_URL,
        source_archive_sha256=sha256_file(archive_path),
        license_id=SAKSPAPIR_LICENSE_ID,
        license_url=SAKSPAPIR_LICENSE_URL,
        extraction_policy={"page_language_code": "nno"},
    )
    loaded_manifest = load_silver_corpus_manifest(manifest_path)

    assert loaded_manifest == written_manifest
    assert loaded_manifest.sentence_count == 3
    assert loaded_manifest.document_count == 2
    assert loaded_manifest.license_id == "CC0-1.0"
    validate_silver_corpus(sentences_path=output_path, manifest=loaded_manifest)


def test_streaming_json_parser_handles_chunk_boundaries() -> None:
    corpus = {
        f"urn:uuid:{index:04d}": [["1", "nno", "Ei setning her. " * 40]]
        for index in range(25)
    }
    payload = json.dumps(corpus, ensure_ascii=False)

    for chunk_character_count in (7, 64, 1024):
        items = dict(
            _iter_json_object_items(
                io.StringIO(payload),
                chunk_character_count=chunk_character_count,
            )
        )
        assert items == corpus


def test_streaming_json_parser_rejects_invalid_documents() -> None:
    with pytest.raises(ValueError, match="top-level JSON object"):
        tuple(_iter_json_object_items(io.StringIO("[]")))

    with pytest.raises(ValueError, match="unterminated value"):
        tuple(_iter_json_object_items(io.StringIO('{"urn": [["1", "nno"')))


def test_sakspapir_rejects_malformed_page_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "sakspapir.tar.gz"
    payload = json.dumps({"urn:uuid:0001": [["1", "nno"]]}).encode()
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("sakspapir_nno_01.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="page_number, language_code, text"):
        tuple(
            iter_sakspapir_silver_sentences(
                archive_path=archive_path,
                extraction_policy=_POLICY,
            )
        )
