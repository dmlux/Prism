"""Typed, provenance-carrying corpora for offline silver-data preparation."""

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from prism.data.examples import PretokenizedSentence


SILVER_CORPUS_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverCorpusManifest:
    format_version: int
    corpus_id: str
    language_tag: str
    source_url: str
    source_archive_sha256: str
    license_id: str
    license_url: str
    sentence_count: int
    token_count: int
    document_count: int
    extraction_policy: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.format_version != SILVER_CORPUS_FORMAT_VERSION:
            raise ValueError("Unsupported silver-corpus format version.")
        for value, label in (
            (self.corpus_id, "corpus ID"),
            (self.language_tag, "language tag"),
            (self.source_url, "source URL"),
            (self.license_id, "license ID"),
            (self.license_url, "license URL"),
        ):
            if not value or value.strip() != value:
                raise ValueError(
                    f"Silver-corpus {label} must be non-empty and trimmed."
                )
        if len(self.source_archive_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_archive_sha256
        ):
            raise ValueError("Silver-corpus archive SHA-256 must be lowercase hex.")
        for count, label in (
            (self.sentence_count, "sentence count"),
            (self.token_count, "token count"),
            (self.document_count, "document count"),
        ):
            if count < 0:
                raise ValueError(f"Silver-corpus {label} must not be negative.")
        if self.sentence_count == 0:
            raise ValueError("Silver corpus must contain at least one sentence.")
        if self.token_count < self.sentence_count:
            raise ValueError(
                "Silver-corpus token count must cover every non-empty sentence."
            )
        if not 1 <= self.document_count <= self.sentence_count:
            raise ValueError(
                "Silver-corpus document count must cover the recorded sentences."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PretokenizedSilverSentence:
    document_id: str
    sentence_index: int
    model_input: PretokenizedSentence

    def __post_init__(self) -> None:
        if not self.document_id or self.document_id.strip() != self.document_id:
            raise ValueError(
                "Silver sentence document ID must be non-empty and trimmed."
            )
        if self.sentence_index < 0:
            raise ValueError("Silver sentence index must not be negative.")


def sentence_fingerprint(sentence: PretokenizedSentence) -> str:
    digest = hashlib.sha256()
    for token in sentence.tokens:
        normalized_token = unicodedata.normalize("NFC", token).casefold()
        digest.update(normalized_token.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pretokenized_silver_corpus(
    *,
    sentences: Iterable[PretokenizedSilverSentence],
    output_path: Path,
    manifest_path: Path,
    corpus_id: str,
    language_tag: str,
    source_url: str,
    source_archive_sha256: str,
    license_id: str,
    license_url: str,
    extraction_policy: Mapping[str, object],
) -> SilverCorpusManifest:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    sentence_count = 0
    token_count = 0
    document_ids: set[str] = set()
    with temporary_output_path.open("w", encoding="utf-8") as output:
        for sentence in sentences:
            output.write(
                json.dumps(
                    {
                        "document_id": sentence.document_id,
                        "sentence_index": sentence.sentence_index,
                        "tokens": sentence.model_input.tokens,
                        "has_space_before": sentence.model_input.has_space_before,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output.write("\n")
            sentence_count += 1
            token_count += len(sentence.model_input.tokens)
            document_ids.add(sentence.document_id)

    try:
        manifest = SilverCorpusManifest(
            format_version=SILVER_CORPUS_FORMAT_VERSION,
            corpus_id=corpus_id,
            language_tag=language_tag,
            source_url=source_url,
            source_archive_sha256=source_archive_sha256,
            license_id=license_id,
            license_url=license_url,
            sentence_count=sentence_count,
            token_count=token_count,
            document_count=len(document_ids),
            extraction_policy=dict(extraction_policy),
        )
    except ValueError:
        temporary_output_path.unlink(missing_ok=True)
        raise
    temporary_manifest_path.write_text(
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_output_path.replace(output_path)
    temporary_manifest_path.replace(manifest_path)
    return manifest


def load_silver_corpus_manifest(path: Path) -> SilverCorpusManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Silver-corpus manifest must contain a JSON object.")
    return SilverCorpusManifest(**value)


def iter_pretokenized_silver_sentences(
    path: Path,
) -> Iterator[PretokenizedSilverSentence]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                document_id = value["document_id"]
                sentence_index = value["sentence_index"]
                tokens = value["tokens"]
                has_space_before = value["has_space_before"]
                if (
                    not isinstance(document_id, str)
                    or not isinstance(sentence_index, int)
                    or isinstance(sentence_index, bool)
                    or not isinstance(tokens, list)
                    or any(not isinstance(token, str) for token in tokens)
                    or not isinstance(has_space_before, list)
                    or any(
                        not isinstance(has_space, bool)
                        for has_space in has_space_before
                    )
                ):
                    raise ValueError("record fields have invalid types")
                yield PretokenizedSilverSentence(
                    document_id=document_id,
                    sentence_index=sentence_index,
                    model_input=PretokenizedSentence(
                        tokens=tuple(tokens),
                        has_space_before=tuple(has_space_before),
                    ),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid silver-corpus record at line {line_number}."
                ) from error


def validate_silver_corpus(
    *,
    sentences_path: Path,
    manifest: SilverCorpusManifest,
) -> None:
    sentence_count = 0
    token_count = 0
    document_ids: set[str] = set()
    for sentence in iter_pretokenized_silver_sentences(sentences_path):
        sentence_count += 1
        token_count += len(sentence.model_input.tokens)
        document_ids.add(sentence.document_id)

    actual_counts = (sentence_count, token_count, len(document_ids))
    expected_counts = (
        manifest.sentence_count,
        manifest.token_count,
        manifest.document_count,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            "Silver-corpus record counts do not match the manifest: "
            f"expected {expected_counts}, got {actual_counts}."
        )
