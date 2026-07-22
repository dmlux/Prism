import pytest

from prism.languages.norwegian import (
    NORWEGIAN_BOKMAAL_PROFILE,
    NORWEGIAN_BOKMAAL_UD_2_17_TREEBANK,
    NORWEGIAN_NYNORSK_PROFILE,
    NORWEGIAN_WRITTEN_STANDARD_PROFILES,
    norwegian_profile_for_language_tag,
    norwegian_training_profiles_for_language_tag,
    norwegian_model_supports_language_tag,
)


def test_norwegian_profile_registry_resolves_written_standards() -> None:
    assert norwegian_profile_for_language_tag("nb") is NORWEGIAN_BOKMAAL_PROFILE
    assert norwegian_profile_for_language_tag("nn") is NORWEGIAN_NYNORSK_PROFILE


def test_norwegian_profile_registry_resolves_pinned_ud_2_17_treebank() -> None:
    profile = norwegian_profile_for_language_tag(
        "nb",
        treebank_release="2.17",
    )

    assert profile.student_backbone is NORWEGIAN_BOKMAAL_PROFILE.student_backbone
    assert profile.gold_treebank is NORWEGIAN_BOKMAAL_UD_2_17_TREEBANK
    assert profile.gold_treebank.revision == "b8618a2b935762d6ccd2dc997180c3e46f74f6b7"


def test_norwegian_profile_registry_rejects_unknown_language_tag() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported Norwegian language tag: de",
    ):
        norwegian_profile_for_language_tag("de")


def test_norwegian_model_family_contains_both_written_standards() -> None:
    assert NORWEGIAN_WRITTEN_STANDARD_PROFILES == (
        NORWEGIAN_BOKMAAL_PROFILE,
        NORWEGIAN_NYNORSK_PROFILE,
    )


def test_norwegian_training_profiles_resolve_joint_model_family() -> None:
    assert norwegian_training_profiles_for_language_tag("nb") == (
        NORWEGIAN_BOKMAAL_PROFILE,
    )
    assert norwegian_training_profiles_for_language_tag("nn") == (
        NORWEGIAN_NYNORSK_PROFILE,
    )
    assert norwegian_training_profiles_for_language_tag("no") == (
        NORWEGIAN_BOKMAAL_PROFILE,
        NORWEGIAN_NYNORSK_PROFILE,
    )


def test_joint_norwegian_model_supports_both_written_standards() -> None:
    assert norwegian_model_supports_language_tag("no", "nb")
    assert norwegian_model_supports_language_tag("no", "nn")

    assert norwegian_model_supports_language_tag("nb", "nb")
    assert not norwegian_model_supports_language_tag("nb", "nn")
    assert not norwegian_model_supports_language_tag("nn", "nb")

    assert norwegian_model_supports_language_tag("no", "no")
    assert not norwegian_model_supports_language_tag("nb", "no")
    assert not norwegian_model_supports_language_tag("nn", "no")
