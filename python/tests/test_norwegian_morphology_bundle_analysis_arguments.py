from pathlib import Path

from prism.languages.norwegian.analyze_morphology_bundles import (
    parse_morphology_bundle_analysis_arguments,
)


def test_parse_morphology_bundle_analysis_arguments() -> None:
    arguments = parse_morphology_bundle_analysis_arguments(
        (
            "--language-tag",
            "nn",
            "--treebank-release",
            "2.17",
            "--analysis",
            "runs/oracle/nn.json",
        )
    )

    assert arguments.language_tag == "nn"
    assert arguments.treebank_release == "2.17"
    assert arguments.analysis_path == Path("runs/oracle/nn.json")


def test_analysis_path_default_follows_language_tag() -> None:
    arguments = parse_morphology_bundle_analysis_arguments(
        ("--language-tag", "nn"),
    )

    assert arguments.analysis_path == Path(
        "runs/morphology-bundles/nn-development-oracle.json"
    )
