from dataclasses import replace
from pathlib import Path

from prism.data.treebanks import UniversalDependenciesTreebankSpec
from prism.languages.norwegian.backbones import (
    NORBERT4_BASE_BACKBONE,
    NORBERT4_LARGE_BACKBONE,
    NORBERT4_XSMALL_BACKBONE,
)
from prism.languages.profile import LanguageProfileSpec

NORWEGIAN_BOKMAAL_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_Norwegian-Bokmaal",
    revision="396d11f0c2bd290a2a2711015c04ac25bc3dcc06",
    license_id="CC-BY-SA-4.0",
    training_path=Path("data/raw/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu"),
    development_path=Path("data/raw/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu"),
    test_path=Path("data/raw/UD_Norwegian-Bokmaal/no_bokmaal-ud-test.conllu"),
)

NORWEGIAN_NYNORSK_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_Norwegian-Nynorsk",
    revision="aaeb9d90c748c2bd9e272f180b599484f9f05ac6",
    license_id="CC-BY-SA-4.0",
    training_path=Path("data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu"),
    development_path=Path("data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu"),
    test_path=Path("data/raw/UD_Norwegian-Nynorsk/no_nynorsk-ud-test.conllu"),
)

NORWEGIAN_BOKMAAL_UD_2_17_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_Norwegian-Bokmaal",
    revision="b8618a2b935762d6ccd2dc997180c3e46f74f6b7",
    license_id="CC-BY-SA-4.0",
    training_path=Path(
        "data/raw/ud-2.17/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu"
    ),
    development_path=Path(
        "data/raw/ud-2.17/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu"
    ),
    test_path=Path("data/raw/ud-2.17/UD_Norwegian-Bokmaal/no_bokmaal-ud-test.conllu"),
)

NORWEGIAN_NYNORSK_UD_2_17_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_Norwegian-Nynorsk",
    revision="2bbe9c67d5e81eadf237b7840ebac31bffca38ae",
    license_id="CC-BY-SA-4.0",
    training_path=Path(
        "data/raw/ud-2.17/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu"
    ),
    development_path=Path(
        "data/raw/ud-2.17/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu"
    ),
    test_path=Path("data/raw/ud-2.17/UD_Norwegian-Nynorsk/no_nynorsk-ud-test.conllu"),
)

NORWEGIAN_BOKMAAL_PROFILE = LanguageProfileSpec(
    language_tag="nb",
    display_name="Norwegian Bokmål",
    student_backbone=NORBERT4_XSMALL_BACKBONE,
    teacher_backbone=NORBERT4_BASE_BACKBONE,
    alternate_teacher_backbones=(NORBERT4_LARGE_BACKBONE,),
    gold_treebank=NORWEGIAN_BOKMAAL_TREEBANK,
)

NORWEGIAN_NYNORSK_PROFILE = LanguageProfileSpec(
    language_tag="nn",
    display_name="Norwegian Nynorsk",
    student_backbone=NORBERT4_XSMALL_BACKBONE,
    teacher_backbone=NORBERT4_BASE_BACKBONE,
    alternate_teacher_backbones=(NORBERT4_LARGE_BACKBONE,),
    gold_treebank=NORWEGIAN_NYNORSK_TREEBANK,
)

NORWEGIAN_WRITTEN_STANDARD_PROFILES = (
    NORWEGIAN_BOKMAAL_PROFILE,
    NORWEGIAN_NYNORSK_PROFILE,
)

_NORWEGIAN_PROFILES_BY_LANGUAGE_TAG = {
    profile.language_tag: profile for profile in NORWEGIAN_WRITTEN_STANDARD_PROFILES
}

_NORWEGIAN_UD_2_17_TREEBANKS_BY_LANGUAGE_TAG = {
    "nb": NORWEGIAN_BOKMAAL_UD_2_17_TREEBANK,
    "nn": NORWEGIAN_NYNORSK_UD_2_17_TREEBANK,
}


def norwegian_profile_for_language_tag(
    language_tag: str,
    *,
    treebank_release: str = "current",
) -> LanguageProfileSpec:
    try:
        profile = _NORWEGIAN_PROFILES_BY_LANGUAGE_TAG[language_tag]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Norwegian language tag: {language_tag}"
        ) from error

    if treebank_release == "current":
        return profile
    if treebank_release == "2.17":
        return replace(
            profile,
            gold_treebank=_NORWEGIAN_UD_2_17_TREEBANKS_BY_LANGUAGE_TAG[language_tag],
        )
    raise ValueError(f"Unsupported Norwegian treebank release: {treebank_release}")


def norwegian_training_profiles_for_language_tag(
    language_tag: str,
    *,
    treebank_release: str = "current",
) -> tuple[LanguageProfileSpec, ...]:
    if language_tag == "no":
        return tuple(
            norwegian_profile_for_language_tag(
                profile.language_tag,
                treebank_release=treebank_release,
            )
            for profile in NORWEGIAN_WRITTEN_STANDARD_PROFILES
        )

    return (
        norwegian_profile_for_language_tag(
            language_tag,
            treebank_release=treebank_release,
        ),
    )


def norwegian_model_supports_language_tag(
    model_language_tag: str,
    language_tag: str,
) -> bool:
    model_profiles = norwegian_training_profiles_for_language_tag(model_language_tag)
    requested_profiles = norwegian_training_profiles_for_language_tag(language_tag)

    return all(profile in model_profiles for profile in requested_profiles)
