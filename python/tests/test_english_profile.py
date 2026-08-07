import pytest

from prism.languages.english import (
    ENGLISH_PROFILE,
    ENGLISH_PROFILES,
    ETTIN_ENCODER_17M_BACKBONE,
    ETTIN_ENCODER_400M_BACKBONE,
    english_model_supports_language_tag,
    english_profile_for_language_tag,
    english_training_profiles_for_language_tag,
)
from prism.languages.english.profile import ENGLISH_EWT_UD_2_17_TREEBANK


def test_english_profile_resolves_en() -> None:
    assert english_profile_for_language_tag("en") is ENGLISH_PROFILE


def test_english_profile_wires_ettin_backbones_and_ewt() -> None:
    assert ENGLISH_PROFILE.language_tag == "en"
    assert ENGLISH_PROFILE.display_name == "English"
    assert ENGLISH_PROFILE.student_backbone is ETTIN_ENCODER_17M_BACKBONE
    assert ENGLISH_PROFILE.teacher_backbone is ETTIN_ENCODER_400M_BACKBONE
    assert ENGLISH_PROFILE.student_backbone.model_id == "jhu-clsp/ettin-encoder-17m"
    assert ENGLISH_PROFILE.teacher_backbone.model_id == "jhu-clsp/ettin-encoder-400m"
    assert (
        ENGLISH_PROFILE.gold_treebank.repository_id
        == "UniversalDependencies/UD_English-EWT"
    )
    assert ENGLISH_PROFILE.gold_treebank.license_id == "CC-BY-SA-4.0"


def test_english_student_backbone_uses_modernbert_export_settings() -> None:
    backbone = ENGLISH_PROFILE.student_backbone
    assert backbone.trust_remote_code is False
    assert backbone.attention_implementation == "eager"
    assert backbone.config_overrides == (("reference_compile", False),)


def test_english_profile_resolves_pinned_ud_2_17_treebank() -> None:
    profile = english_profile_for_language_tag("en", treebank_release="2.17")

    assert profile.student_backbone is ENGLISH_PROFILE.student_backbone
    assert profile.gold_treebank is ENGLISH_EWT_UD_2_17_TREEBANK
    assert profile.gold_treebank.revision == "c5baffde1e106bcd828c520109eb905bfc3ac06f"


def test_english_profile_rejects_unknown_language_tag() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported English language tag: de",
    ):
        english_profile_for_language_tag("de")


def test_english_model_family_is_a_single_profile() -> None:
    assert ENGLISH_PROFILES == (ENGLISH_PROFILE,)
    assert english_training_profiles_for_language_tag("en") == (ENGLISH_PROFILE,)


def test_english_model_supports_its_own_language_tag() -> None:
    assert english_model_supports_language_tag("en", "en")
