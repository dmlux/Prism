"""Norwegian Prism language profiles."""

from prism.languages.norwegian.backbones import (
    NORBERT4_BASE_BACKBONE,
    NORBERT4_LARGE_BACKBONE,
    NORBERT4_XSMALL_BACKBONE,
)
from prism.languages.norwegian.profile import (
    NORWEGIAN_BOKMAAL_PROFILE,
    NORWEGIAN_BOKMAAL_UD_2_17_TREEBANK,
    NORWEGIAN_NYNORSK_PROFILE,
    NORWEGIAN_NYNORSK_UD_2_17_TREEBANK,
    NORWEGIAN_WRITTEN_STANDARD_PROFILES,
    norwegian_model_supports_language_tag,
    norwegian_profile_for_language_tag,
    norwegian_training_profiles_for_language_tag,
)

__all__ = [
    "NORBERT4_BASE_BACKBONE",
    "NORBERT4_LARGE_BACKBONE",
    "NORBERT4_XSMALL_BACKBONE",
    "NORWEGIAN_BOKMAAL_PROFILE",
    "NORWEGIAN_BOKMAAL_UD_2_17_TREEBANK",
    "NORWEGIAN_NYNORSK_PROFILE",
    "NORWEGIAN_NYNORSK_UD_2_17_TREEBANK",
    "NORWEGIAN_WRITTEN_STANDARD_PROFILES",
    "norwegian_profile_for_language_tag",
    "norwegian_training_profiles_for_language_tag",
    "norwegian_model_supports_language_tag",
]
