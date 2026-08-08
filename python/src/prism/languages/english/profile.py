from dataclasses import replace
from pathlib import Path

from prism.data.treebanks import UniversalDependenciesTreebankSpec
from prism.languages.english.backbones import (
    ETTIN_ENCODER_17M_BACKBONE,
    ETTIN_ENCODER_400M_BACKBONE,
)
from prism.languages.profile import LanguageProfileSpec

# UD_English-EWT is the largest English UD treebank (~254k tokens) and, unlike
# UD_English-GUM (CC BY-NC-SA 4.0), ships under CC-BY-SA-4.0 — matching Prism's
# existing gold-data licensing story.
ENGLISH_EWT_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_English-EWT",
    revision="4a4d77f599ea53cc405f85d0cec4b2f14f81d42b",
    license_id="CC-BY-SA-4.0",
    training_path=Path("data/raw/UD_English-EWT/en_ewt-ud-train.conllu"),
    development_path=Path("data/raw/UD_English-EWT/en_ewt-ud-dev.conllu"),
    test_path=Path("data/raw/UD_English-EWT/en_ewt-ud-test.conllu"),
)

ENGLISH_EWT_UD_2_17_TREEBANK = UniversalDependenciesTreebankSpec(
    repository_id="UniversalDependencies/UD_English-EWT",
    revision="c5baffde1e106bcd828c520109eb905bfc3ac06f",
    license_id="CC-BY-SA-4.0",
    training_path=Path("data/raw/ud-2.17/UD_English-EWT/en_ewt-ud-train.conllu"),
    development_path=Path("data/raw/ud-2.17/UD_English-EWT/en_ewt-ud-dev.conllu"),
    test_path=Path("data/raw/ud-2.17/UD_English-EWT/en_ewt-ud-test.conllu"),
)

ENGLISH_PROFILE = LanguageProfileSpec(
    language_tag="en",
    display_name="English",
    student_backbone=ETTIN_ENCODER_17M_BACKBONE,
    teacher_backbone=ETTIN_ENCODER_400M_BACKBONE,
    gold_treebank=ENGLISH_EWT_TREEBANK,
    # Ettin is ModernBERT; its int8 PT2E path differs from GPT-BERT/NorBERT4.
    quantization="modernbert",
)

ENGLISH_PROFILES = (ENGLISH_PROFILE,)

_ENGLISH_PROFILES_BY_LANGUAGE_TAG = {
    profile.language_tag: profile for profile in ENGLISH_PROFILES
}

_ENGLISH_UD_2_17_TREEBANKS_BY_LANGUAGE_TAG = {
    "en": ENGLISH_EWT_UD_2_17_TREEBANK,
}


def english_profile_for_language_tag(
    language_tag: str,
    *,
    treebank_release: str = "current",
) -> LanguageProfileSpec:
    try:
        profile = _ENGLISH_PROFILES_BY_LANGUAGE_TAG[language_tag]
    except KeyError as error:
        raise ValueError(
            f"Unsupported English language tag: {language_tag}"
        ) from error

    if treebank_release == "current":
        return profile
    if treebank_release == "2.17":
        return replace(
            profile,
            gold_treebank=_ENGLISH_UD_2_17_TREEBANKS_BY_LANGUAGE_TAG[language_tag],
        )
    raise ValueError(f"Unsupported English treebank release: {treebank_release}")


def english_training_profiles_for_language_tag(
    language_tag: str,
    *,
    treebank_release: str = "current",
) -> tuple[LanguageProfileSpec, ...]:
    return (
        english_profile_for_language_tag(
            language_tag,
            treebank_release=treebank_release,
        ),
    )


def english_model_supports_language_tag(
    model_language_tag: str,
    language_tag: str,
) -> bool:
    model_profiles = english_training_profiles_for_language_tag(model_language_tag)
    requested_profiles = english_training_profiles_for_language_tag(language_tag)

    return all(profile in model_profiles for profile in requested_profiles)
