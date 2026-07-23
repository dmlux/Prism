import io
import tarfile
import unicodedata
from pathlib import Path

import pytest

from prism.data import (
    NBDIGITAL_CORPUS_ID,
    NBDIGITAL_LICENSE_ID,
    NBDIGITAL_LICENSE_URL,
    NBDIGITAL_SOURCE_URL,
    iter_nbdigital_silver_sentences,
    iter_pretokenized_silver_sentences,
    load_silver_corpus_manifest,
    parse_nbdigital_document_name,
    sentence_fingerprint,
    sha256_file,
    validate_silver_corpus,
    write_pretokenized_silver_corpus,
)
from prism.data.examples import PretokenizedSentence


def _write_archive(path: Path) -> None:
    documents = {
        ("corpus/digibok_2007022801063-1991-nob-975--Eksempel.txt.xml"): (
            '<doc><w l="hei" p="interj" c="">Hei</w>'
            '<w l="$." p="clb" c="&lt;&lt;&lt; &lt;punkt&gt;">.</w>'
            '<w l="ny" p="adj" c="">Ny</w>'
            '<w l="setning" p="subst" c="">setning</w>'
            '<w l="$!" p="clb" c="&lt;&lt;&lt; &lt;utrop&gt;">!</w></doc>'
        ),
        ("corpus/digibok_2007022801064-1992-nob-800--Lav_kvalitet.txt.xml"): (
            '<doc><w l="ignorer" p="verb" c="">Ignorer</w>'
            '<w l="$." p="clb" c="&lt;&lt;&lt; &lt;punkt&gt;">.</w></doc>'
        ),
    }
    with tarfile.open(path, mode="w:gz") as archive:
        for name, text in documents.items():
            value = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))


def test_nbdigital_parser_filters_ocr_gold_overlap_and_duplicates(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "corpus.tar.gz"
    _write_archive(archive_path)
    metadata = parse_nbdigital_document_name(
        "digibok_2007022801063-1991-nob-975--Eksempel.txt.xml"
    )

    all_sentences = tuple(
        iter_nbdigital_silver_sentences(
            archive_path=archive_path,
            minimum_ocr_confidence=0.95,
            maximum_token_count=8,
        )
    )
    excluded_fingerprint = sentence_fingerprint(all_sentences[0].model_input)
    filtered_sentences = tuple(
        iter_nbdigital_silver_sentences(
            archive_path=archive_path,
            minimum_ocr_confidence=0.95,
            maximum_token_count=8,
            excluded_sentence_fingerprints=(excluded_fingerprint,),
        )
    )

    assert metadata.publication_year == 1991
    assert metadata.ocr_confidence == 0.975
    assert metadata.title == "Eksempel"
    assert tuple(sentence.model_input.tokens for sentence in all_sentences) == (
        ("Hei", "."),
        ("Ny", "setning", "!"),
    )
    assert all_sentences[0].model_input.has_space_before == (False, False)
    assert tuple(sentence.model_input.tokens for sentence in filtered_sentences) == (
        ("Ny", "setning", "!"),
    )


def test_silver_corpus_manifest_and_records_round_trip(tmp_path: Path) -> None:
    archive_path = tmp_path / "corpus.tar.gz"
    _write_archive(archive_path)
    sentences = iter_nbdigital_silver_sentences(
        archive_path=archive_path,
        minimum_ocr_confidence=0.95,
        maximum_token_count=8,
    )
    output_path = tmp_path / "sentences.jsonl"
    manifest_path = tmp_path / "manifest.json"

    written_manifest = write_pretokenized_silver_corpus(
        sentences=sentences,
        output_path=output_path,
        manifest_path=manifest_path,
        corpus_id=NBDIGITAL_CORPUS_ID,
        language_tag="nb",
        source_url=NBDIGITAL_SOURCE_URL,
        source_archive_sha256=sha256_file(archive_path),
        license_id=NBDIGITAL_LICENSE_ID,
        license_url=NBDIGITAL_LICENSE_URL,
        extraction_policy={"minimum_ocr_confidence": 0.95},
    )
    loaded_manifest = load_silver_corpus_manifest(manifest_path)
    loaded_sentences = tuple(iter_pretokenized_silver_sentences(output_path))

    assert loaded_manifest == written_manifest
    assert loaded_manifest.sentence_count == 2
    assert loaded_manifest.token_count == 5
    assert loaded_manifest.document_count == 1
    assert loaded_manifest.license_id == "CC0-1.0"
    assert len(loaded_sentences) == 2
    validate_silver_corpus(
        sentences_path=output_path,
        manifest=loaded_manifest,
    )


def test_sentence_fingerprint_normalizes_unicode_and_case() -> None:
    composed = PretokenizedSentence(
        tokens=("Å",),
        has_space_before=(False,),
    )
    decomposed = PretokenizedSentence(
        tokens=(unicodedata.normalize("NFD", "å"),),
        has_space_before=(False,),
    )

    assert sentence_fingerprint(composed) == sentence_fingerprint(decomposed)


def test_silver_corpus_rejects_empty_output(tmp_path: Path) -> None:
    output_path = tmp_path / "sentences.jsonl"

    with pytest.raises(
        ValueError,
        match="must contain at least one sentence",
    ):
        write_pretokenized_silver_corpus(
            sentences=(),
            output_path=output_path,
            manifest_path=tmp_path / "manifest.json",
            corpus_id=NBDIGITAL_CORPUS_ID,
            language_tag="nb",
            source_url=NBDIGITAL_SOURCE_URL,
            source_archive_sha256="0" * 64,
            license_id=NBDIGITAL_LICENSE_ID,
            license_url=NBDIGITAL_LICENSE_URL,
            extraction_policy={},
        )

    assert not output_path.exists()
