"""Public-domain silver-source ingestion for Project Gutenberg plain text.

Project Gutenberg distributes public-domain literary works as plain-text files
wrapped in a Project Gutenberg header and footer. This adapter strips that PG
boilerplate — so only the underlying public-domain work remains, free of the
Project Gutenberg trademark license — reflows the hard-wrapped body into
paragraphs, and runs the shared rule-based sentence extraction from
``prism.data.segmentation``.

Including both British and American authors broadens dialect coverage cheaply;
the adapter performs no spelling normalization, so British and American
orthography reach the student verbatim (see ``docs/MODEL_STRATEGY.md``).

The input is either a directory tree of ``.txt`` files or a ``.tar``/
``.tar.gz`` archive of them; the document id is taken from each file's stem
(e.g. ``pg1342.txt`` -> ``pg1342``).
"""

import re
import tarfile
from collections.abc import Collection, Iterator
from pathlib import Path, PurePosixPath

from prism.data.segmentation import (
    SentenceExtractionPolicy,
    extract_pretokenized_sentences,
)
from prism.data.silver import PretokenizedSilverSentence, sentence_fingerprint


GUTENBERG_CORPUS_ID = "project-gutenberg"
GUTENBERG_LANGUAGE_TAG = "en"
GUTENBERG_SOURCE_URL = "https://www.gutenberg.org/"
GUTENBERG_LICENSE_ID = "PublicDomain"
GUTENBERG_LICENSE_URL = "https://creativecommons.org/publicdomain/mark/1.0/"
GUTENBERG_EXTRACTION_VERSION = "prism-gutenberg-plain-v1"

_START_MARKER = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_END_MARKER = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)
_PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n")
# Lines that are decorative or structural rather than prose.
_SKIPPED_PARAGRAPH_MARKERS = ("_", "*", "=")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return only the work body, without the PG header and footer.

    When the markers are absent (older or hand-edited files) the whole text is
    returned; the paragraph and sentence filters then carry the quality load.
    """

    start = _START_MARKER.search(text)
    body = text[start.end() :] if start is not None else text
    end = _END_MARKER.search(body)
    if end is not None:
        body = body[: end.start()]
    return body


def gutenberg_plain_paragraphs(body: str) -> Iterator[str]:
    """Yield reflowed prose paragraphs of one work body as plain text.

    Project Gutenberg hard-wraps lines at roughly seventy columns; blank lines
    separate paragraphs. Each blank-line-delimited block is reflowed into one
    line. All-caps or decorated blocks (chapter headings, transcriber notes)
    are skipped conservatively.
    """

    for block in _PARAGRAPH_SEPARATOR.split(body):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        paragraph = " ".join(lines)
        if paragraph[0] in _SKIPPED_PARAGRAPH_MARKERS:
            continue
        letters = [character for character in paragraph if character.isalpha()]
        if letters and all(character.isupper() for character in letters):
            # Chapter headings and title blocks are fully upper-case.
            continue
        yield paragraph


def _document_id(name: str) -> str:
    stem = PurePosixPath(name).name
    if stem.endswith(".txt"):
        stem = stem[: -len(".txt")]
    return stem


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _iter_documents(archive_path: Path) -> Iterator[tuple[str, str]]:
    if archive_path.is_dir():
        for path in sorted(archive_path.rglob("*.txt")):
            yield _document_id(path.name), _decode_text(path.read_bytes())
        return
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".txt")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            file = archive.extractfile(member)
            if file is None:
                raise ValueError(f"Cannot read Gutenberg member: {member.name!r}")
            yield _document_id(member.name), _decode_text(file.read())


def iter_gutenberg_silver_sentences(
    *,
    archive_path: Path,
    extraction_policy: SentenceExtractionPolicy,
    excluded_sentence_fingerprints: Collection[str] = (),
) -> Iterator[PretokenizedSilverSentence]:
    seen_fingerprints: set[str] = set()
    excluded_fingerprints = set(excluded_sentence_fingerprints)
    for document_id, text in _iter_documents(archive_path):
        body = strip_gutenberg_boilerplate(text)
        sentence_index = 0
        for paragraph in gutenberg_plain_paragraphs(body):
            for model_input in extract_pretokenized_sentences(
                paragraph,
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
