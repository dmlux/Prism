"""English Prism language profile."""

from prism.languages.english.backbones import (
    ETTIN_ENCODER_17M_BACKBONE,
    ETTIN_ENCODER_400M_BACKBONE,
)
from prism.languages.english.profile import (
    ENGLISH_EWT_UD_2_17_TREEBANK,
    ENGLISH_PROFILE,
    ENGLISH_PROFILES,
    english_model_supports_language_tag,
    english_profile_for_language_tag,
    english_training_profiles_for_language_tag,
)

__all__ = [
    "ETTIN_ENCODER_17M_BACKBONE",
    "ETTIN_ENCODER_400M_BACKBONE",
    "ENGLISH_EWT_UD_2_17_TREEBANK",
    "ENGLISH_PROFILE",
    "ENGLISH_PROFILES",
    "english_profile_for_language_tag",
    "english_training_profiles_for_language_tag",
    "english_model_supports_language_tag",
]
