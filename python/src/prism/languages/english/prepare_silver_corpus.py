"""Prepare a provenance-carrying, teacher-ready English silver source.

Two permissively licensed sources, chosen to broaden dialect coverage:
Project Gutenberg (public-domain literary English, both British and American
authors) and the English Wikipedia dump (CC BY-SA 4.0). The adapters perform
no spelling normalization, so British and American orthography reach the
student verbatim (see ``docs/MODEL_STRATEGY.md``).
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from prism.conllu import Token, read_sentences
from prism.data import (
    GUTENBERG_CORPUS_ID,
    GUTENBERG_EXTRACTION_VERSION,
    GUTENBERG_LANGUAGE_TAG,
    GUTENBERG_LICENSE_ID,
    GUTENBERG_LICENSE_URL,
    GUTENBERG_SOURCE_URL,
    SENTENCE_EXTRACTION_POLICY_VERSION,
    WIKIPEDIA_ENG_CORPUS_ID,
    WIKIPEDIA_ENG_LANGUAGE_TAG,
    WIKIPEDIA_ENG_SOURCE_URL,
    WIKIPEDIA_LICENSE_ID,
    WIKIPEDIA_LICENSE_URL,
    WIKITEXT_EXTRACTION_VERSION,
    PretokenizedSentence,
    iter_gutenberg_silver_sentences,
    iter_wikipedia_silver_sentences,
    sentence_fingerprint,
    sha256_file,
    write_pretokenized_silver_corpus,
)
from prism.languages.english import (
    english_training_profiles_for_language_tag,
)
from prism.languages.english.silver_extraction import (
    ENGLISH_MINIMUM_LETTER_TOKEN_RATIO,
    ENGLISH_MINIMUM_SILVER_TOKEN_COUNT,
    ENGLISH_SILVER_ABBREVIATIONS,
    english_sentence_extraction_policy,
)


GUTENBERG_SOURCE = "gutenberg-eng"
WIKIPEDIA_SOURCE = "wikipedia-eng"
_DEFAULT_OUTPUT_DIRECTORIES = {
    GUTENBERG_SOURCE: Path("data/processed/gutenberg-eng"),
    WIKIPEDIA_SOURCE: Path("data/processed/wikipedia-eng"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverCorpusPreparationArguments:
    source: str
    archive_path: Path
    output_path: Path
    manifest_path: Path
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
    for profile in english_training_profiles_for_language_tag(
        "en",
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
            "Prepare a permissively licensed English silver corpus "
            "for teacher labeling."
        ),
    )
    parser.add_argument(
        "--source",
        choices=(GUTENBERG_SOURCE, WIKIPEDIA_SOURCE),
        default=GUTENBERG_SOURCE,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help=(
            "Gutenberg: a directory tree of .txt files or a .tar/.tar.gz "
            "archive of them. Wikipedia: the pages-articles XML(.bz2) dump."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
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

    output_directory = _DEFAULT_OUTPUT_DIRECTORIES[parsed.source]
    output_path = parsed.output or output_directory / "pretokenized.jsonl"
    manifest_path = parsed.manifest or output_directory / "manifest.json"
    return SilverCorpusPreparationArguments(
        source=parsed.source,
        archive_path=parsed.archive,
        output_path=output_path,
        manifest_path=manifest_path,
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
        "sentence_boundary_source": SENTENCE_EXTRACTION_POLICY_VERSION,
        "minimum_token_count": ENGLISH_MINIMUM_SILVER_TOKEN_COUNT,
        "minimum_letter_token_ratio": ENGLISH_MINIMUM_LETTER_TOKEN_RATIO,
        "abbreviation_tokens": tuple(sorted(ENGLISH_SILVER_ABBREVIATIONS)),
    }
    if arguments.source == GUTENBERG_SOURCE:
        sentences = iter_gutenberg_silver_sentences(
            archive_path=arguments.archive_path,
            extraction_policy=english_sentence_extraction_policy(
                maximum_token_count=arguments.maximum_token_count,
            ),
            excluded_sentence_fingerprints=excluded_fingerprints,
        )
        corpus_id = GUTENBERG_CORPUS_ID
        language_tag = GUTENBERG_LANGUAGE_TAG
        source_url = GUTENBERG_SOURCE_URL
        license_id = GUTENBERG_LICENSE_ID
        license_url = GUTENBERG_LICENSE_URL
        extraction_policy = {
            **shared_policy,
            "markup_removal": GUTENBERG_EXTRACTION_VERSION,
        }
    else:
        sentences = iter_wikipedia_silver_sentences(
            archive_path=arguments.archive_path,
            extraction_policy=english_sentence_extraction_policy(
                maximum_token_count=arguments.maximum_token_count,
            ),
            excluded_sentence_fingerprints=excluded_fingerprints,
        )
        corpus_id = WIKIPEDIA_ENG_CORPUS_ID
        language_tag = WIKIPEDIA_ENG_LANGUAGE_TAG
        source_url = WIKIPEDIA_ENG_SOURCE_URL
        license_id = WIKIPEDIA_LICENSE_ID
        license_url = WIKIPEDIA_LICENSE_URL
        extraction_policy = {
            **shared_policy,
            "markup_removal": WIKITEXT_EXTRACTION_VERSION,
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
