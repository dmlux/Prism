"""Prepare a provenance-carrying, teacher-ready Norwegian silver source."""

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
    SAKSPAPIR_CORPUS_ID,
    SAKSPAPIR_LANGUAGE_TAG,
    SAKSPAPIR_LICENSE_ID,
    SAKSPAPIR_LICENSE_URL,
    SAKSPAPIR_PAGE_LANGUAGE_CODE,
    SAKSPAPIR_SOURCE_URL,
    SENTENCE_EXTRACTION_POLICY_VERSION,
    WIKIPEDIA_LICENSE_ID,
    WIKIPEDIA_LICENSE_URL,
    WIKIPEDIA_NNO_CORPUS_ID,
    WIKIPEDIA_NNO_LANGUAGE_TAG,
    WIKIPEDIA_NNO_SOURCE_URL,
    WIKITEXT_EXTRACTION_VERSION,
    PretokenizedSentence,
    iter_nbdigital_silver_sentences,
    iter_sakspapir_silver_sentences,
    iter_wikipedia_silver_sentences,
    sentence_fingerprint,
    sha256_file,
    write_pretokenized_silver_corpus,
)
from prism.languages.norwegian import (
    norwegian_training_profiles_for_language_tag,
)
from prism.languages.norwegian.silver_extraction import (
    NORWEGIAN_MINIMUM_LETTER_TOKEN_RATIO,
    NORWEGIAN_MINIMUM_SILVER_TOKEN_COUNT,
    NORWEGIAN_SILVER_ABBREVIATIONS,
    norwegian_sentence_extraction_policy,
)


NBDIGITAL_SOURCE = "nbdigital-nob"
SAKSPAPIR_SOURCE = "sakspapir-nno"
WIKIPEDIA_SOURCE = "wikipedia-nno"
_DEFAULT_MINIMUM_OCR_CONFIDENCE = 0.95
_DEFAULT_OUTPUT_DIRECTORIES = {
    NBDIGITAL_SOURCE: Path("data/processed/nbdigital-nob-free"),
    SAKSPAPIR_SOURCE: Path("data/processed/sakspapir-nno"),
    WIKIPEDIA_SOURCE: Path("data/processed/wikipedia-nno"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverCorpusPreparationArguments:
    source: str
    archive_path: Path
    output_path: Path
    manifest_path: Path
    minimum_ocr_confidence: float | None
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
        description=(
            "Prepare a permissively licensed Norwegian silver corpus "
            "for teacher labeling."
        ),
    )
    parser.add_argument(
        "--source",
        choices=(NBDIGITAL_SOURCE, SAKSPAPIR_SOURCE, WIKIPEDIA_SOURCE),
        default=NBDIGITAL_SOURCE,
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--minimum-ocr-confidence",
        type=float,
        default=None,
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
    if parsed.maximum_token_count <= 0:
        parser.error("--maximum-token-count must be greater than zero")

    minimum_ocr_confidence = parsed.minimum_ocr_confidence
    if parsed.source == NBDIGITAL_SOURCE:
        if minimum_ocr_confidence is None:
            minimum_ocr_confidence = _DEFAULT_MINIMUM_OCR_CONFIDENCE
        if not 0.0 <= minimum_ocr_confidence <= 1.0:
            parser.error("--minimum-ocr-confidence must be between zero and one")
    elif minimum_ocr_confidence is not None:
        parser.error(
            "--minimum-ocr-confidence only applies to the nbdigital-nob source"
        )

    output_directory = _DEFAULT_OUTPUT_DIRECTORIES[parsed.source]
    output_path = parsed.output or output_directory / "pretokenized.jsonl"
    manifest_path = parsed.manifest or output_directory / "manifest.json"
    return SilverCorpusPreparationArguments(
        source=parsed.source,
        archive_path=parsed.archive,
        output_path=output_path,
        manifest_path=manifest_path,
        minimum_ocr_confidence=minimum_ocr_confidence,
        maximum_token_count=parsed.maximum_token_count,
        treebank_release=parsed.treebank_release,
    )


def main() -> None:
    arguments = parse_preparation_arguments()
    archive_sha256 = sha256_file(arguments.archive_path)
    excluded_fingerprints = _gold_sentence_fingerprints(
        treebank_release=arguments.treebank_release
    )
    shared_policy = {
        "maximum_token_count": arguments.maximum_token_count,
        "source_tags_used_as_targets": False,
        "deduplication": "sha256-casefolded-token-sequence",
        "excluded_gold_splits": ("train", "development", "test"),
        "treebank_release": arguments.treebank_release,
        "excluded_gold_fingerprint_count": len(excluded_fingerprints),
    }
    if arguments.source == NBDIGITAL_SOURCE:
        assert arguments.minimum_ocr_confidence is not None
        sentences = iter_nbdigital_silver_sentences(
            archive_path=arguments.archive_path,
            minimum_ocr_confidence=arguments.minimum_ocr_confidence,
            maximum_token_count=arguments.maximum_token_count,
            excluded_sentence_fingerprints=excluded_fingerprints,
        )
        corpus_id = NBDIGITAL_CORPUS_ID
        language_tag = NBDIGITAL_LANGUAGE_TAG
        source_url = NBDIGITAL_SOURCE_URL
        license_id = NBDIGITAL_LICENSE_ID
        license_url = NBDIGITAL_LICENSE_URL
        extraction_policy = {
            **shared_policy,
            "minimum_ocr_confidence": arguments.minimum_ocr_confidence,
            "sentence_boundary_source": "oslo-bergen-tagger-<<<",
        }
    elif arguments.source == WIKIPEDIA_SOURCE:
        sentences = iter_wikipedia_silver_sentences(
            archive_path=arguments.archive_path,
            extraction_policy=norwegian_sentence_extraction_policy(
                maximum_token_count=arguments.maximum_token_count,
            ),
            excluded_sentence_fingerprints=excluded_fingerprints,
        )
        corpus_id = WIKIPEDIA_NNO_CORPUS_ID
        language_tag = WIKIPEDIA_NNO_LANGUAGE_TAG
        source_url = WIKIPEDIA_NNO_SOURCE_URL
        license_id = WIKIPEDIA_LICENSE_ID
        license_url = WIKIPEDIA_LICENSE_URL
        extraction_policy = {
            **shared_policy,
            "markup_removal": WIKITEXT_EXTRACTION_VERSION,
            "sentence_boundary_source": SENTENCE_EXTRACTION_POLICY_VERSION,
            "minimum_token_count": NORWEGIAN_MINIMUM_SILVER_TOKEN_COUNT,
            "minimum_letter_token_ratio": NORWEGIAN_MINIMUM_LETTER_TOKEN_RATIO,
            "abbreviation_tokens": tuple(sorted(NORWEGIAN_SILVER_ABBREVIATIONS)),
        }
    else:
        sentences = iter_sakspapir_silver_sentences(
            archive_path=arguments.archive_path,
            extraction_policy=norwegian_sentence_extraction_policy(
                maximum_token_count=arguments.maximum_token_count,
            ),
            excluded_sentence_fingerprints=excluded_fingerprints,
        )
        corpus_id = SAKSPAPIR_CORPUS_ID
        language_tag = SAKSPAPIR_LANGUAGE_TAG
        source_url = SAKSPAPIR_SOURCE_URL
        license_id = SAKSPAPIR_LICENSE_ID
        license_url = SAKSPAPIR_LICENSE_URL
        extraction_policy = {
            **shared_policy,
            "page_language_code": SAKSPAPIR_PAGE_LANGUAGE_CODE,
            "sentence_boundary_source": SENTENCE_EXTRACTION_POLICY_VERSION,
            "minimum_token_count": NORWEGIAN_MINIMUM_SILVER_TOKEN_COUNT,
            "minimum_letter_token_ratio": NORWEGIAN_MINIMUM_LETTER_TOKEN_RATIO,
            "abbreviation_tokens": tuple(sorted(NORWEGIAN_SILVER_ABBREVIATIONS)),
        }
    manifest = write_pretokenized_silver_corpus(
        sentences=sentences,
        output_path=arguments.output_path,
        manifest_path=arguments.manifest_path,
        corpus_id=corpus_id,
        language_tag=language_tag,
        source_url=source_url,
        source_archive_sha256=archive_sha256,
        license_id=license_id,
        license_url=license_url,
        extraction_policy=extraction_policy,
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
