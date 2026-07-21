from torch import nn

from prism.modeling.alignment import TokenPoolingStrategy, align_subwords_to_tokens
from prism.modeling.batches import TokenizedBatch
from prism.modeling.character_batches import CharacterTokenBatch
from prism.modeling.character_encoders import CharacterCnnTokenEncoder
from prism.modeling.encoders import contextualize_subwords
from prism.modeling.heads import TokenTaskHeadArchitecture, TokenTaskHeads
from prism.modeling.layer_aggregation import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)
from prism.modeling.outputs import TokenTaskLogits
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

    def forward(
        self,
        batch: TokenizedBatch,
        character_batch: CharacterTokenBatch | None = None,
    ) -> TokenTaskLogits:
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

        character_hidden_states = None
        if self.character_encoder is not None:
            if character_batch is None:
                raise ValueError("Character-aware tagger requires character inputs.")
            if character_batch.token_mask.shape != batch.token_mask.shape:
                raise ValueError("Character and tokenizer token dimensions must match.")
            character_hidden_states = self.character_encoder(character_batch)
        elif character_batch is not None:
            raise ValueError("Character inputs require a character-aware tagger.")

        return self.heads(
            token_batch.hidden_states,
            character_hidden_states=character_hidden_states,
        )


def build_pretrained_token_tagger(
    *,
    backbone_spec: PretrainedBackboneSpec,
    schema: TokenTaskSchema,
    dropout_probability: float,
    pooling_strategy: TokenPoolingStrategy = TokenPoolingStrategy.FIRST,
    head_architecture: TokenTaskHeadArchitecture = TokenTaskHeadArchitecture.LINEAR,
    layer_aggregation_strategy: BackboneLayerAggregationStrategy = (
        BackboneLayerAggregationStrategy.LAST
    ),
    character_vocabulary_size: int | None = None,
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
