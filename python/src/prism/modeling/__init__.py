"""Transformer model inputs, components, and outputs."""

from prism.modeling.batches import TokenizedBatch
from prism.modeling.backbones import (
    PretrainedBackboneSpec,
    load_backbone_model,
)
from prism.modeling.tokenizers import (
    load_backbone_tokenizer,
    prepare_pretokenized_words,
    tokenize_pretokenized_sentences,
)
from prism.modeling.alignment import (
    align_subwords_to_tokens,
    build_padded_token_alignment,
    find_first_subword_indices,
)
from prism.modeling.outputs import (
    ContextualizedSubwordBatch,
    ContextualizedTokenBatch,
    TokenTaskLogits,
)
from prism.modeling.encoders import contextualize_subwords
from prism.modeling.heads import TokenClassificationHead, TokenTaskHeads
from prism.modeling.taggers import (
    TokenTagger,
    build_pretrained_token_tagger,
)

__all__ = [
    "TokenizedBatch",
    "PretrainedBackboneSpec",
    "load_backbone_model",
    "load_backbone_tokenizer",
    "prepare_pretokenized_words",
    "tokenize_pretokenized_sentences",
    "align_subwords_to_tokens",
    "build_padded_token_alignment",
    "find_first_subword_indices",
    "ContextualizedSubwordBatch",
    "ContextualizedTokenBatch",
    "TokenTaskLogits",
    "contextualize_subwords",
    "TokenClassificationHead",
    "TokenTaskHeads",
    "TokenTagger",
    "build_pretrained_token_tagger",
]
