"""Measure training-derived morphology-bundle candidate coverage."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from prism.conllu import Token, read_sentences
from prism.evaluation import (
    MorphologyBundleCoverage,
    MorphologyBundleExample,
    MorphologyBundleInventory,
    evaluate_morphology_bundle_oracle,
    morphology_bundle_from_features,
)
from prism.languages.norwegian import (
    norwegian_profile_for_language_tag,
    norwegian_training_profiles_for_language_tag,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleAnalysisArguments:
    language_tag: str
    treebank_release: str
    analysis_path: Path


def parse_morphology_bundle_analysis_arguments(
    arguments: Sequence[str] | None = None,
) -> MorphologyBundleAnalysisArguments:
    parser = argparse.ArgumentParser(
        description="Analyze Norwegian morphology-bundle candidate coverage.",
    )
    parser.add_argument("--language-tag", choices=("nb", "nn"), default="nb")
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        dest="analysis_path",
    )
    parsed = parser.parse_args(arguments)
    analysis_path = parsed.analysis_path or Path(
        f"runs/morphology-bundles/{parsed.language_tag}-development-oracle.json"
    )
    return MorphologyBundleAnalysisArguments(
        language_tag=parsed.language_tag,
        treebank_release=parsed.treebank_release,
        analysis_path=analysis_path,
    )


def _examples(tokens: Sequence[Sequence[Token]]) -> tuple[MorphologyBundleExample, ...]:
    return tuple(
        MorphologyBundleExample(
            upos=token.upos,
            bundle=morphology_bundle_from_features(token.features),
        )
        for sentence in tokens
        for token in sentence
    )


def _print_coverage(name: str, coverage: MorphologyBundleCoverage) -> None:
    print()
    print(name)
    print(f"Tokens                  {coverage.token_count:>7}")
    print(f"Global inventory        {coverage.global_coverage:>10.6f}")
    print(f"Gold-UPOS inventory     {coverage.gold_upos_coverage:>10.6f}")
    for top_k in coverage.top_k:
        print(
            f"Gold-UPOS top-{top_k.candidate_count:<2}       "
            f"{top_k.covered_token_count / coverage.token_count:>10.6f}"
        )


def main() -> None:
    arguments = parse_morphology_bundle_analysis_arguments()
    training_profiles = norwegian_training_profiles_for_language_tag(
        "no",
        treebank_release=arguments.treebank_release,
    )
    evaluation_profile = norwegian_profile_for_language_tag(
        arguments.language_tag,
        treebank_release=arguments.treebank_release,
    )
    training_tokens = tuple(
        sentence
        for profile in training_profiles
        for sentence in read_sentences(profile.gold_treebank.training_path)
    )
    development_tokens = read_sentences(
        evaluation_profile.gold_treebank.development_path
    )
    inventory = MorphologyBundleInventory.from_examples(_examples(training_tokens))
    metrics = evaluate_morphology_bundle_oracle(
        inventory=inventory,
        development_examples=_examples(development_tokens),
    )

    print("Language tag:", arguments.language_tag)
    print("Treebank release:", arguments.treebank_release)
    print("Training tokens:", metrics.training_example_count)
    print("Distinct bundles:", metrics.distinct_bundle_count)
    print("Distinct UPOS-bundle pairs:", metrics.distinct_upos_bundle_count)
    print("Maximum candidates per UPOS:", metrics.maximum_candidate_count)
    _print_coverage("ALL TOKENS", metrics.overall)
    _print_coverage("ANNOTATED TOKENS", metrics.annotated)

    arguments.analysis_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.analysis_path.write_text(
        json.dumps(
            {
                "language_tag": arguments.language_tag,
                "treebank_release": arguments.treebank_release,
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print("Analysis:", arguments.analysis_path)


if __name__ == "__main__":
    main()
