from pathlib import Path

from prism.data.treebanks import UniversalDependenciesTreebankSpec
from prism.languages.norwegian.backbones import (
    NORBERT4_BASE_BACKBONE,
    NORBERT4_XSMALL_BACKBONE,
)
from prism.languages.profile import LanguageProfileSpec

NORWEGIAN_BOKMAAL_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_Norwegian-Bokmaal",
    revision="396d11f0c2bd290a2a2711015c04ac25bc3dcc06",
    license_id="CC-BY-SA-4.0",
    training_path=Path("data/raw/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu"),
    development_path=Path("data/raw/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu"),
)

NORWEGIAN_NYNORSK_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_Norwegian-Nynorsk",
    revision="aaeb9d90c748c2bd9e272f180b599484f9f05ac6",
    license_id="CC-BY-SA-4.0",
    training_path=Path("data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu"),
    development_path=Path("data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu"),
)

NORWEGIAN_BOKMAAL_PROFILE = LanguageProfileSpec(
    language_tag="nb",
    display_name="Norwegian Bokmål",
    student_backbone=NORBERT4_XSMALL_BACKBONE,
    teacher_backbone=NORBERT4_BASE_BACKBONE,
    gold_treebank=NORWEGIAN_BOKMAAL_TREEBANK,
)

NORWEGIAN_NYNORSK_PROFILE = LanguageProfileSpec(
    language_tag="nn",
    display_name="Norwegian Nynorsk",
    student_backbone=NORBERT4_XSMALL_BACKBONE,
    teacher_backbone=NORBERT4_BASE_BACKBONE,
    gold_treebank=NORWEGIAN_NYNORSK_TREEBANK,
)

NORWEGIAN_WRITTEN_STANDARD_PROFILES = (
    NORWEGIAN_BOKMAAL_PROFILE,
    NORWEGIAN_NYNORSK_PROFILE,
)

_NORWEGIAN_PROFILES_BY_LANGUAGE_TAG = {
    profile.language_tag: profile for profile in NORWEGIAN_WRITTEN_STANDARD_PROFILES
}


def norwegian_profile_for_language_tag(
    language_tag: str,
) -> LanguageProfileSpec:
    try:
        return _NORWEGIAN_PROFILES_BY_LANGUAGE_TAG[language_tag]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Norwegian language tag: {language_tag}"
        ) from error


def norwegian_training_profiles_for_language_tag(
    language_tag: str,
) -> tuple[LanguageProfileSpec, ...]:
    if language_tag == "no":
        return NORWEGIAN_WRITTEN_STANDARD_PROFILES

    return (norwegian_profile_for_language_tag(language_tag),)


def norwegian_model_supports_language_tag(
    model_language_tag: str,
    language_tag: str,
) -> bool:
    model_profiles = norwegian_training_profiles_for_language_tag(model_language_tag)
    requested_profiles = norwegian_training_profiles_for_language_tag(language_tag)

    return all(profile in model_profiles for profile in requested_profiles)
