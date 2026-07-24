"""Transformer model inputs, components, and outputs."""

from prism.modeling.batches import TokenizedBatch
from prism.modeling.character_batches import (
    CharacterTokenBatch,
    encode_character_token_batch,
)
from prism.modeling.character_encoders import (
    CharacterCnnTokenEncoder,
    CharacterResidualFusion,
)
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
    TokenPoolingStrategy,
    align_subwords_to_tokens,
    build_padded_token_alignment,
    find_subword_spans,
)
from prism.modeling.outputs import (
    ContextualizedSubwordBatch,
    ContextualizedTokenBatch,
    TokenTaskHiddenStates,
    TokenTaskLogits,
)
from prism.modeling.encoders import contextualize_subwords
from prism.modeling.layer_aggregation import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)
from prism.modeling.heads import (
    MorphologyPreHeadArchitecture,
    SharedResidualTokenProjection,
    TaskResidualAdapter,
    TokenClassificationHead,
    TokenTaskHeadArchitecture,
    TokenTaskHeads,
    WideSharedResidualTokenProjection,
)
from prism.modeling.structured_morphology import StructuredMorphologyDecoder
from prism.modeling.morphology_agreement import (
    MorphologyAgreementRefiner,
    MorphologyAgreementRefinerSpec,
    validate_morphology_agreement_refiner_spec,
)
from prism.modeling.morphology_bundle_reranker import (
    CompositionalMorphologyBundleScorer,
    MorphologyBundleCandidate,
    MorphologyBundleLossGradientScope,
    MorphologyBundleReranker,
    MorphologyBundleRerankerSpec,
    MorphologyBundleScorerArchitecture,
    validate_morphology_bundle_reranker_spec,
)
from prism.modeling.decoding import (
    MorphologyLogitCorrection,
    apply_morphology_feature_logit_correction,
    apply_morphology_logit_correction,
    decode_morphology_feature_logits,
)
from prism.modeling.taggers import (
    TokenTagger,
    build_pretrained_token_tagger,
)

__all__ = [
    "TokenizedBatch",
    "CharacterTokenBatch",
    "encode_character_token_batch",
    "CharacterCnnTokenEncoder",
    "CharacterResidualFusion",
    "PretrainedBackboneSpec",
    "load_backbone_model",
    "load_backbone_tokenizer",
    "prepare_pretokenized_words",
    "tokenize_pretokenized_sentences",
    "TokenPoolingStrategy",
    "align_subwords_to_tokens",
    "build_padded_token_alignment",
    "find_subword_spans",
    "ContextualizedSubwordBatch",
    "ContextualizedTokenBatch",
    "TokenTaskHiddenStates",
    "TokenTaskLogits",
    "contextualize_subwords",
    "BackboneLayerAggregation",
    "BackboneLayerAggregationStrategy",
    "TokenClassificationHead",
    "MorphologyPreHeadArchitecture",
    "TokenTaskHeadArchitecture",
    "TokenTaskHeads",
    "SharedResidualTokenProjection",
    "TaskResidualAdapter",
    "WideSharedResidualTokenProjection",
    "StructuredMorphologyDecoder",
    "MorphologyAgreementRefiner",
    "MorphologyAgreementRefinerSpec",
    "validate_morphology_agreement_refiner_spec",
    "CompositionalMorphologyBundleScorer",
    "MorphologyBundleCandidate",
    "MorphologyBundleLossGradientScope",
    "MorphologyBundleReranker",
    "MorphologyBundleRerankerSpec",
    "MorphologyBundleScorerArchitecture",
    "validate_morphology_bundle_reranker_spec",
    "MorphologyLogitCorrection",
    "apply_morphology_feature_logit_correction",
    "apply_morphology_logit_correction",
    "decode_morphology_feature_logits",
    "TokenTagger",
    "build_pretrained_token_tagger",
]
