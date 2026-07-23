"""Pretokenized CC0 silver-source ingestion for the NBdigital corpus."""

import io
import math
import re
import tarfile
import xml.etree.ElementTree as ElementTree
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from prism.data.examples import PretokenizedSentence
from prism.data.silver import PretokenizedSilverSentence, sentence_fingerprint


NBDIGITAL_CORPUS_ID = "oai:nb.no:sbr-43"
NBDIGITAL_LANGUAGE_TAG = "nb"
NBDIGITAL_SOURCE_URL = "https://www.nb.no/sbfil/tekst/20160229_posdata-nob-free.tar.gz"
NBDIGITAL_LICENSE_ID = "CC0-1.0"
NBDIGITAL_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

_DOCUMENT_NAME_PATTERN = re.compile(
    r"^(?P<document_id>digibok_[^-]+)-"
    r"(?P<publication_year>[0-9]{4})-"
    r"(?P<language_code>[a-z]{3})-"
    r"(?P<ocr_confidence>[0-9]{3})--"
    r"(?P<title>.+)\.txt\.xml$"
)
_ATTACH_TO_PREVIOUS = frozenset(".,;:!?…)]}»")


@dataclass(frozen=True, slots=True, kw_only=True)
class NbDigitalDocumentMetadata:
    document_id: str
    publication_year: int
    language_code: str
    ocr_confidence: float
    title: str


def parse_nbdigital_document_name(name: str) -> NbDigitalDocumentMetadata:
    match = _DOCUMENT_NAME_PATTERN.fullmatch(PurePosixPath(name).name)
    if match is None:
        raise ValueError(f"Unsupported NBdigital document name: {name!r}")
    return NbDigitalDocumentMetadata(
        document_id=match.group("document_id"),
        publication_year=int(match.group("publication_year")),
        language_code=match.group("language_code"),
        ocr_confidence=int(match.group("ocr_confidence")) / 1000.0,
        title=match.group("title").replace("_", " "),
    )


def _has_space_before(tokens: list[str]) -> tuple[bool, ...]:
    return tuple(
        False if index == 0 or (token and token[0] in _ATTACH_TO_PREVIOUS) else True
        for index, token in enumerate(tokens)
    )


def _iter_document_sentences(
    *,
    xml_bytes: bytes,
    document_id: str,
    maximum_token_count: int,
) -> Iterator[PretokenizedSilverSentence]:
    tokens: list[str] = []
    sentence_index = 0
    for _, element in ElementTree.iterparse(io.BytesIO(xml_bytes), events=("end",)):
        if element.tag != "w":
            continue
        token = "" if element.text is None else element.text.strip()
        categories = element.attrib.get("c", "")
        element.clear()
        if token:
            tokens.append(token)
        if "<<<" not in categories:
            continue
        if tokens and len(tokens) <= maximum_token_count:
            yield PretokenizedSilverSentence(
                document_id=document_id,
                sentence_index=sentence_index,
                model_input=PretokenizedSentence(
                    tokens=tuple(tokens),
                    has_space_before=_has_space_before(tokens),
                ),
            )
        sentence_index += 1
        tokens = []

    if tokens and len(tokens) <= maximum_token_count:
        yield PretokenizedSilverSentence(
            document_id=document_id,
            sentence_index=sentence_index,
            model_input=PretokenizedSentence(
                tokens=tuple(tokens),
                has_space_before=_has_space_before(tokens),
            ),
        )


def iter_nbdigital_silver_sentences(
    *,
    archive_path: Path,
    minimum_ocr_confidence: float,
    maximum_token_count: int,
    excluded_sentence_fingerprints: Collection[str] = (),
) -> Iterator[PretokenizedSilverSentence]:
    if (
        not math.isfinite(minimum_ocr_confidence)
        or not 0.0 <= minimum_ocr_confidence <= 1.0
    ):
        raise ValueError("Minimum OCR confidence must be between zero and one.")
    if maximum_token_count <= 0:
        raise ValueError("Maximum token count must be positive.")

    seen_fingerprints: set[str] = set()
    excluded_fingerprints = set(excluded_sentence_fingerprints)
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".txt.xml")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            try:
                metadata = parse_nbdigital_document_name(member.name)
            except ValueError:
                continue
            if metadata.language_code != "nob":
                continue
            if metadata.ocr_confidence < minimum_ocr_confidence:
                continue
            file = archive.extractfile(member)
            if file is None:
                raise ValueError(f"Cannot read NBdigital member: {member.name!r}")
            for sentence in _iter_document_sentences(
                xml_bytes=file.read(),
                document_id=metadata.document_id,
                maximum_token_count=maximum_token_count,
            ):
                fingerprint = sentence_fingerprint(sentence.model_input)
                if (
                    fingerprint in excluded_fingerprints
                    or fingerprint in seen_fingerprints
                ):
                    continue
                seen_fingerprints.add(fingerprint)
                yield sentence
