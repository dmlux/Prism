"""Dataset-specific preprocessing contracts."""

from prism.data.norwegian_bokmaal import (
    normalize_norwegian_bokmaal_ud_lemma,
    encode_norwegian_bokmaal_sentence,
    encode_norwegian_bokmaal_sentences,
    build_norwegian_bokmaal_schema,
)
from prism.data.examples import (
    SupervisedSentence,
    SupervisedCorpus,
    TokenTargets,
)

__all__ = [
    "normalize_norwegian_bokmaal_ud_lemma",
    "encode_norwegian_bokmaal_sentence",
    "encode_norwegian_bokmaal_sentences",
    "SupervisedSentence",
    "SupervisedCorpus",
    "TokenTargets",
    "build_norwegian_bokmaal_schema",
]
