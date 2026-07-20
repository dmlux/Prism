import pytest

from prism.languages.norwegian import (
    NORWEGIAN_BOKMAAL_PROFILE,
    NORWEGIAN_NYNORSK_PROFILE,
    NORWEGIAN_WRITTEN_STANDARD_PROFILES,
    norwegian_profile_for_language_tag,
)


def test_norwegian_profile_registry_resolves_written_standards() -> None:
    assert norwegian_profile_for_language_tag("nb") is NORWEGIAN_BOKMAAL_PROFILE
    assert norwegian_profile_for_language_tag("nn") is NORWEGIAN_NYNORSK_PROFILE


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
