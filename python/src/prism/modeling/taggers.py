import torch
from torch import nn

from prism.modeling.alignment import TokenPoolingStrategy, align_subwords_to_tokens
from prism.modeling.batches import TokenizedBatch
from prism.modeling.character_batches import CharacterTokenBatch
from prism.modeling.character_encoders import CharacterCnnTokenEncoder
from prism.modeling.encoders import contextualize_subwords
from prism.modeling.heads import (
    MorphologyPreHeadArchitecture,
    TokenTaskHeadArchitecture,
    TokenTaskHeads,
)
from prism.modeling.layer_aggregation import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)
from prism.modeling.outputs import TokenTaskHiddenStates, TokenTaskLogits
from prism.modeling.morphology_bundle_reranker import MorphologyBundleRerankerSpec
from prism.schema import TokenTaskSchema
from prism.modeling.backbones import (
    PretrainedBackboneSpec,
    load_backbone_model,
)


class TokenTagger(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        heads: TokenTaskHeads,
        pooling_strategy: TokenPoolingStrategy = TokenPoolingStrategy.FIRST,
        layer_aggregation: BackboneLayerAggregation | None = None,
        character_encoder: CharacterCnnTokenEncoder | None = None,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.heads = heads
        self.pooling_strategy = pooling_strategy
        self.layer_aggregation = layer_aggregation or BackboneLayerAggregation(
            strategy=BackboneLayerAggregationStrategy.LAST
        )
        self.character_encoder = character_encoder

    def encode_pooled_token_states(
        self,
        batch: TokenizedBatch,
    ) -> "torch.Tensor":
        """Word-aligned backbone representations before any task head.

        This boundary is shared by every tagger size and tokenizer, so it is
        the comparison point for token-relation distillation.
        """

        subword_batch = contextualize_subwords(
            model=self.backbone,
            batch=batch,
            layer_aggregation=self.layer_aggregation,
        )
        token_batch = align_subwords_to_tokens(
            subword_batch=subword_batch,
            tokenized_batch=batch,
            pooling_strategy=self.pooling_strategy,
        )
        return token_batch.hidden_states

    def _character_hidden_states(
        self,
        batch: TokenizedBatch,
        character_batch: CharacterTokenBatch | None,
    ) -> "torch.Tensor | None":
        if self.character_encoder is not None:
            if character_batch is None:
                raise ValueError("Character-aware tagger requires character inputs.")
            if character_batch.token_mask.shape != batch.token_mask.shape:
                raise ValueError("Character and tokenizer token dimensions must match.")
            return self.character_encoder(character_batch)
        if character_batch is not None:
            raise ValueError("Character inputs require a character-aware tagger.")
        return None

    def encode_task_hidden_states(
        self,
        batch: TokenizedBatch,
        character_batch: CharacterTokenBatch | None = None,
    ) -> TokenTaskHiddenStates:
        return self.heads.encode_hidden_states(
            self.encode_pooled_token_states(batch),
            character_hidden_states=self._character_hidden_states(
                batch,
                character_batch,
            ),
        )

    def forward_with_pooled_states(
        self,
        batch: TokenizedBatch,
        character_batch: CharacterTokenBatch | None = None,
    ) -> "tuple[torch.Tensor, TokenTaskLogits]":
        """Classify while also returning the pooled backbone states.

        Token-relation distillation needs both in one backbone pass.
        """

        pooled_states = self.encode_pooled_token_states(batch)
        task_hidden_states = self.heads.encode_hidden_states(
            pooled_states,
            character_hidden_states=self._character_hidden_states(
                batch,
                character_batch,
            ),
        )
        return pooled_states, self.heads.classify_hidden_states(task_hidden_states)

    def forward(
        self,
        batch: TokenizedBatch,
        character_batch: CharacterTokenBatch | None = None,
    ) -> TokenTaskLogits:
        task_hidden_states = self.encode_task_hidden_states(
            batch,
            character_batch=character_batch,
        )
        return self.heads.classify_hidden_states(task_hidden_states)


def build_pretrained_token_tagger(
    *,
    backbone_spec: PretrainedBackboneSpec,
    schema: TokenTaskSchema,
    dropout_probability: float,
    pooling_strategy: TokenPoolingStrategy = TokenPoolingStrategy.FIRST,
    head_architecture: TokenTaskHeadArchitecture = TokenTaskHeadArchitecture.LINEAR,
    morphology_pre_head_architecture: MorphologyPreHeadArchitecture = (
        MorphologyPreHeadArchitecture.IDENTITY
    ),
    layer_aggregation_strategy: BackboneLayerAggregationStrategy = (
        BackboneLayerAggregationStrategy.LAST
    ),
    character_vocabulary_size: int | None = None,
    morphology_bundle_reranker_spec: MorphologyBundleRerankerSpec | None = None,
) -> TokenTagger:
    backbone = load_backbone_model(backbone_spec)
    hidden_size = getattr(
        backbone.config,
        "hidden_size",
        None,
    )

    if (
        not isinstance(hidden_size, int)
        or isinstance(hidden_size, bool)
        or hidden_size <= 0
    ):
        raise ValueError("Backbone configuration must provide a positive hidden size.")

    heads = TokenTaskHeads(
        hidden_size=hidden_size,
        schema=schema,
        dropout_probability=dropout_probability,
        architecture=head_architecture,
        morphology_pre_head_architecture=morphology_pre_head_architecture,
        morphology_bundle_reranker_spec=morphology_bundle_reranker_spec,
    )
    character_encoder = None
    if head_architecture.uses_character_encoder:
        if character_vocabulary_size is None:
            raise ValueError(
                "Character-aware architecture requires a character vocabulary."
            )
        character_encoder = CharacterCnnTokenEncoder(
            vocabulary_size=character_vocabulary_size,
            hidden_size=hidden_size,
        )
    elif character_vocabulary_size is not None:
        raise ValueError(
            "Character vocabulary requires a character-aware architecture."
        )

    return TokenTagger(
        backbone=backbone,
        heads=heads,
        pooling_strategy=pooling_strategy,
        layer_aggregation=BackboneLayerAggregation(
            strategy=layer_aggregation_strategy,
        ),
        character_encoder=character_encoder,
    )
