"""Versioned, runtime-independent model schemas."""

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
    "MORPHOLOGY_SCHEMA_VERSION",
    "NO_MORPHOLOGY_VALUE",
    "MorphologyFeatureSchema",
    "MorphologySchema",
    "build_lemma_rule_schema",
    "build_morphology_schema",
    "build_upos_schema",
    "decode_morphology_values",
    "encode_morphology_targets",
    "LEMMA_RULE_SCHEMA_VERSION",
    "LemmaEditRule",
    "LemmaRuleSchema",
    "derive_lemma_edit_rule",
    "UPOS_SCHEMA_VERSION",
    "UposSchema",
    "TokenTaskSchema",
]
