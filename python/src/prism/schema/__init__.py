"""Versioned, runtime-independent model schemas."""

from prism.schema.characters import (
    CHARACTER_END_ID,
    CHARACTER_FIRST_LITERAL_ID,
    CHARACTER_PADDING_ID,
    CHARACTER_START_ID,
    CHARACTER_TRUNCATION_ID,
    CHARACTER_UNKNOWN_ID,
    CHARACTER_VOCABULARY_SCHEMA_VERSION,
    CharacterVocabularySchema,
    build_character_vocabulary_schema,
    normalize_character_token,
)

from prism.schema.morphology import (
    MORPHOLOGY_SCHEMA_VERSION,
    NO_MORPHOLOGY_VALUE,
    MorphologyFeatureSchema,
    MorphologySchema,
)
from prism.schema.builders import (
    build_lemma_rule_schema,
    build_morphology_schema,
    build_upos_schema,
)
from prism.schema.targets import (
    decode_morphology_values,
    encode_morphology_targets,
)
from prism.schema.lemma import (
    LEMMA_RULE_SCHEMA_VERSION,
    LemmaEditRule,
    LemmaRuleSchema,
    derive_lemma_edit_rule,
)
from prism.schema.upos import (
    UPOS_SCHEMA_VERSION,
    UposSchema,
)
from prism.schema.token_tasks import TokenTaskSchema

__all__ = [
    "CHARACTER_END_ID",
    "CHARACTER_FIRST_LITERAL_ID",
    "CHARACTER_PADDING_ID",
    "CHARACTER_START_ID",
    "CHARACTER_TRUNCATION_ID",
    "CHARACTER_UNKNOWN_ID",
    "CHARACTER_VOCABULARY_SCHEMA_VERSION",
    "CharacterVocabularySchema",
    "MORPHOLOGY_SCHEMA_VERSION",
    "NO_MORPHOLOGY_VALUE",
    "MorphologyFeatureSchema",
    "MorphologySchema",
    "build_lemma_rule_schema",
    "build_character_vocabulary_schema",
    "build_morphology_schema",
    "build_upos_schema",
    "decode_morphology_values",
    "encode_morphology_targets",
    "normalize_character_token",
    "LEMMA_RULE_SCHEMA_VERSION",
    "LemmaEditRule",
    "LemmaRuleSchema",
    "derive_lemma_edit_rule",
    "UPOS_SCHEMA_VERSION",
    "UposSchema",
    "TokenTaskSchema",
]
