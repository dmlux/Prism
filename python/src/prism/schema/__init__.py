"""Versioned, runtime-independent model schemas."""

from prism.schema.morphology import (
    MORPHOLOGY_SCHEMA_VERSION,
    NO_MORPHOLOGY_VALUE,
    MorphologyFeatureSchema,
    MorphologySchema,
)
from prism.schema.builders import build_morphology_schema
from prism.schema.targets import (
    decode_morphology_values,
    encode_morphology_targets,
)
from prism.schema.lemma import (
    LemmaEditRule,
    derive_lemma_edit_rule,
)

__all__ = [
    "MORPHOLOGY_SCHEMA_VERSION",
    "NO_MORPHOLOGY_VALUE",
    "MorphologyFeatureSchema",
    "MorphologySchema",
    "build_morphology_schema",
    "decode_morphology_values",
    "encode_morphology_targets",
    "LemmaEditRule",
    "derive_lemma_edit_rule",
]