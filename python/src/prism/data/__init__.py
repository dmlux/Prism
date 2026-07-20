"""Dataset-specific preprocessing contracts."""

from prism.data.batches import (
    TokenTaskTargetBatch,
    build_token_task_target_batch,
)
from prism.data.examples import (
    PretokenizedSentence,
    SupervisedCorpus,
    SupervisedSentence,
    TokenTargets,
)
from prism.data.norwegian import (
    build_norwegian_schema,
    encode_norwegian_sentence,
    encode_norwegian_sentences,
    normalize_norwegian_ud_lemma,
)

__all__ = [
    "normalize_norwegian_ud_lemma",
    "encode_norwegian_sentence",
    "encode_norwegian_sentences",
    "PretokenizedSentence",
    "SupervisedSentence",
    "SupervisedCorpus",
    "TokenTargets",
    "build_norwegian_schema",
    "TokenTaskTargetBatch",
    "build_token_task_target_batch",
]
