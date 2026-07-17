"""Dataset-specific preprocessing contracts."""

from prism.data.norwegian_bokmaal import (
    normalize_norwegian_bokmaal_ud_lemma,
    encode_norwegian_bokmaal_sentence,
    encode_norwegian_bokmaal_sentences,
    build_norwegian_bokmaal_schema,
)
from prism.data.examples import (
    PretokenizedSentence,
    SupervisedSentence,
    SupervisedCorpus,
    TokenTargets,
)
from prism.data.batches import (
    TokenTaskTargetBatch,
    build_token_task_target_batch,
)

__all__ = [
    "normalize_norwegian_bokmaal_ud_lemma",
    "encode_norwegian_bokmaal_sentence",
    "encode_norwegian_bokmaal_sentences",
    "PretokenizedSentence",
    "SupervisedSentence",
    "SupervisedCorpus",
    "TokenTargets",
    "build_norwegian_bokmaal_schema",
    "TokenTaskTargetBatch",
    "build_token_task_target_batch",
]
