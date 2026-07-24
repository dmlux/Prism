"""Raw-text CC0 silver-source ingestion for the Nynorsk sakspapir corpus.

Språkbanken resource ``oai:nb.no:sbr-60`` contains roughly 50,000 municipal
documents whose pages are already classified by language. The archive holds a
single large JSON object mapping a document URN to a list of
``[page_number, language_code, page_text]`` entries. Unlike the NBdigital
Bokmål source, the pages carry no word segmentation, sentence boundaries, or
OCR confidence, so this adapter combines a streaming JSON reader with the
conservative rule-based sentence extraction from ``prism.data.segmentation``.
"""

import io
import json
import tarfile
from collections.abc import Collection, Iterator
from pathlib import Path

from prism.data.segmentation import (
    SentenceExtractionPolicy,
    extract_pretokenized_sentences,
)
from prism.data.silver import PretokenizedSilverSentence, sentence_fingerprint


SAKSPAPIR_CORPUS_ID = "oai:nb.no:sbr-60"
SAKSPAPIR_LANGUAGE_TAG = "nn"
SAKSPAPIR_SOURCE_URL = (
    "https://www.nb.no/sbfil/tekst/sakspapir_nno/sakspapir_nno_01.tar.gz"
)
SAKSPAPIR_LICENSE_ID = "CC0-1.0"
SAKSPAPIR_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
SAKSPAPIR_PAGE_LANGUAGE_CODE = "nno"

_READ_CHUNK_CHARACTER_COUNT = 1 << 20


def _iter_json_object_items(
    stream: io.TextIOBase,
    *,
    chunk_character_count: int = _READ_CHUNK_CHARACTER_COUNT,
) -> Iterator[tuple[str, object]]:
    """Stream the top-level key/value pairs of one large JSON object.

    The corpus file is a single JSON object of roughly 900 MB. Loading it with
    ``json.load`` would hold every document in memory at once, so this parser
    keeps only one document value in the buffer at a time.
    """

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0

    def fill() -> bool:
        nonlocal buffer, position
        chunk = stream.read(chunk_character_count)
        if not chunk:
            return False
        buffer = buffer[position:] + chunk
        position = 0
        return True

    def skip_whitespace() -> None:
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or not fill():
                return

    def decode_value() -> object:
        nonlocal position
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if not fill():
                    raise ValueError(
                        "Sakspapir JSON ended inside an unterminated value."
                    ) from None
                continue
            position = end
            return value

    skip_whitespace()
    if position >= len(buffer) or buffer[position] != "{":
        raise ValueError("Sakspapir corpus must contain one top-level JSON object.")
    position += 1

    while True:
        skip_whitespace()
        if position >= len(buffer):
            raise ValueError("Sakspapir JSON ended before the closing brace.")
        if buffer[position] == "}":
            return
        if buffer[position] == ",":
            position += 1
            skip_whitespace()
        key = decode_value()
        if not isinstance(key, str):
            raise ValueError("Sakspapir document keys must be strings.")
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != ":":
            raise ValueError(f"Sakspapir document {key!r} is missing its value.")
        position += 1
        skip_whitespace()
        yield key, decode_value()


def _document_pages(
    document_id: str,
    value: object,
    *,
    page_language_code: str,
) -> Iterator[str]:
    """Yield the matching-language page texts of one document in page order."""

    if not isinstance(value, list):
        raise ValueError(f"Sakspapir document {document_id!r} must be a list.")
    pages: list[tuple[int, str]] = []
    for entry in value:
        if (
            not isinstance(entry, list)
            or len(entry) != 3
            or not all(isinstance(field, str) for field in entry)
        ):
            raise ValueError(
                f"Sakspapir document {document_id!r} must contain "
                "[page_number, language_code, text] entries."
            )
        page_number, language_code, text = entry
        if language_code != page_language_code:
            continue
        try:
            page_order = int(page_number)
        except ValueError as error:
            raise ValueError(
                f"Sakspapir document {document_id!r} has a non-numeric "
                f"page number: {page_number!r}."
            ) from error
        pages.append((page_order, text))
    for _, text in sorted(pages, key=lambda page: page[0]):
        yield text


def iter_sakspapir_silver_sentences(
    *,
    archive_path: Path,
    extraction_policy: SentenceExtractionPolicy,
    page_language_code: str = SAKSPAPIR_PAGE_LANGUAGE_CODE,
    excluded_sentence_fingerprints: Collection[str] = (),
) -> Iterator[PretokenizedSilverSentence]:
    seen_fingerprints: set[str] = set()
    excluded_fingerprints = set(excluded_sentence_fingerprints)
    with tarfile.open(archive_path, mode="r:*") as archive:
        json_members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".json")
            ),
            key=lambda member: member.name,
        )
        if not json_members:
            raise ValueError(
                f"Sakspapir archive contains no JSON corpus member: {archive_path}"
            )
        for member in json_members:
            file = archive.extractfile(member)
            if file is None:
                raise ValueError(f"Cannot read sakspapir member: {member.name!r}")
            text_stream = io.TextIOWrapper(file, encoding="utf-8")
            for document_id, value in _iter_json_object_items(text_stream):
                sentence_index = 0
                for page_text in _document_pages(
                    document_id,
                    value,
                    page_language_code=page_language_code,
                ):
                    for model_input in extract_pretokenized_sentences(
                        page_text,
                        extraction_policy,
                    ):
                        fingerprint = sentence_fingerprint(model_input)
                        if (
                            fingerprint in excluded_fingerprints
                            or fingerprint in seen_fingerprints
                        ):
                            continue
                        seen_fingerprints.add(fingerprint)
                        yield PretokenizedSilverSentence(
                            document_id=document_id,
                            sentence_index=sentence_index,
                            model_input=model_input,
                        )
                        sentence_index += 1
