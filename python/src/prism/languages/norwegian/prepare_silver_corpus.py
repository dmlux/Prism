"""Prepare a provenance-carrying, teacher-ready NBdigital silver source."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from prism.conllu import Token, read_sentences
from prism.data import (
    NBDIGITAL_CORPUS_ID,
    NBDIGITAL_LANGUAGE_TAG,
    NBDIGITAL_LICENSE_ID,
    NBDIGITAL_LICENSE_URL,
    NBDIGITAL_SOURCE_URL,
    PretokenizedSentence,
    iter_nbdigital_silver_sentences,
    sentence_fingerprint,
    sha256_file,
    write_pretokenized_silver_corpus,
)
from prism.languages.norwegian import (
    norwegian_training_profiles_for_language_tag,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverCorpusPreparationArguments:
    archive_path: Path
    output_path: Path
    manifest_path: Path
    minimum_ocr_confidence: float
    maximum_token_count: int
    treebank_release: str


def _pretokenized_sentence_from_ud(
    sentence: Sequence[Token],
) -> PretokenizedSentence:
    tokens = tuple(token.text for token in sentence)
    space_after = tuple(token.space_after for token in sentence)
    return PretokenizedSentence(
        tokens=tokens,
        has_space_before=tuple(
            False if index == 0 else space_after[index - 1]
            for index in range(len(tokens))
        ),
    )


def _gold_sentence_fingerprints(*, treebank_release: str) -> frozenset[str]:
    fingerprints: set[str] = set()
    for profile in norwegian_training_profiles_for_language_tag(
        "no",
        treebank_release=treebank_release,
    ):
        treebank = profile.gold_treebank
        paths = (
            treebank.training_path,
            treebank.development_path,
            treebank.test_path,
        )
        for path in paths:
            if path is None:
                raise ValueError("Gold treebank must expose all three split paths.")
            fingerprints.update(
                sentence_fingerprint(_pretokenized_sentence_from_ud(sentence))
                for sentence in read_sentences(path)
            )
    return frozenset(fingerprints)


def parse_preparation_arguments(
    arguments: Sequence[str] | None = None,
) -> SilverCorpusPreparationArguments:
    parser = argparse.ArgumentParser(
        description="Prepare the CC0 NBdigital Bokmål corpus for teacher labeling.",
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/nbdigital-nob-free/pretokenized.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/nbdigital-nob-free/manifest.json"),
    )
    parser.add_argument(
        "--minimum-ocr-confidence",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--maximum-token-count",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parsed = parser.parse_args(arguments)
    if not 0.0 <= parsed.minimum_ocr_confidence <= 1.0:
        parser.error("--minimum-ocr-confidence must be between zero and one")
    if parsed.maximum_token_count <= 0:
        parser.error("--maximum-token-count must be greater than zero")
    return SilverCorpusPreparationArguments(
        archive_path=parsed.archive,
        output_path=parsed.output,
        manifest_path=parsed.manifest,
        minimum_ocr_confidence=parsed.minimum_ocr_confidence,
        maximum_token_count=parsed.maximum_token_count,
        treebank_release=parsed.treebank_release,
    )


def main() -> None:
    arguments = parse_preparation_arguments()
    archive_sha256 = sha256_file(arguments.archive_path)
    excluded_fingerprints = _gold_sentence_fingerprints(
        treebank_release=arguments.treebank_release
    )
    manifest = write_pretokenized_silver_corpus(
        sentences=iter_nbdigital_silver_sentences(
            archive_path=arguments.archive_path,
            minimum_ocr_confidence=arguments.minimum_ocr_confidence,
            maximum_token_count=arguments.maximum_token_count,
            excluded_sentence_fingerprints=excluded_fingerprints,
        ),
        output_path=arguments.output_path,
        manifest_path=arguments.manifest_path,
        corpus_id=NBDIGITAL_CORPUS_ID,
        language_tag=NBDIGITAL_LANGUAGE_TAG,
        source_url=NBDIGITAL_SOURCE_URL,
        source_archive_sha256=archive_sha256,
        license_id=NBDIGITAL_LICENSE_ID,
        license_url=NBDIGITAL_LICENSE_URL,
        extraction_policy={
            "minimum_ocr_confidence": arguments.minimum_ocr_confidence,
            "maximum_token_count": arguments.maximum_token_count,
            "sentence_boundary_source": "oslo-bergen-tagger-<<<",
            "source_tags_used_as_targets": False,
            "deduplication": "sha256-casefolded-token-sequence",
            "excluded_gold_splits": ("train", "development", "test"),
            "treebank_release": arguments.treebank_release,
            "excluded_gold_fingerprint_count": len(excluded_fingerprints),
        },
    )
    print("Silver source:", manifest.corpus_id)
    print("License:", manifest.license_id)
    print("Documents:", manifest.document_count)
    print("Sentences:", manifest.sentence_count)
    print("Tokens:", manifest.token_count)
    print("Archive SHA-256:", manifest.source_archive_sha256)
    print("Corpus:", arguments.output_path)
    print("Manifest:", arguments.manifest_path)


if __name__ == "__main__":
    main()
