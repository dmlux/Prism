from pathlib import Path

from prism.languages.norwegian import (
    NORBERT4_XSMALL_BACKBONE,
    NORWEGIAN_NYNORSK_PROFILE,
)


def test_norwegian_nynorsk_profile_uses_shared_norwegian_backbone() -> None:
    assert NORWEGIAN_NYNORSK_PROFILE.language_tag == "nn"
    assert NORWEGIAN_NYNORSK_PROFILE.display_name == "Norwegian Nynorsk"
    assert NORWEGIAN_NYNORSK_PROFILE.student_backbone is NORBERT4_XSMALL_BACKBONE

    treebank = NORWEGIAN_NYNORSK_PROFILE.gold_treebank

    assert treebank.repository_id == "UniversalDependencies/UD_Norwegian-Nynorsk"
    assert treebank.revision == ("aaeb9d90c748c2bd9e272f180b599484f9f05ac6")
    assert treebank.license_id == "CC-BY-SA-4.0"
    assert treebank.training_path == Path(
        "data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu"
    )
    assert treebank.development_path == Path(
        "data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu"
    )
