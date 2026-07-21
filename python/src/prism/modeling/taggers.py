from torch import nn

from prism.modeling.alignment import TokenPoolingStrategy, align_subwords_to_tokens
from prism.modeling.batches import TokenizedBatch
from prism.modeling.encoders import contextualize_subwords
from prism.modeling.heads import TokenTaskHeads
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
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.heads = heads
        self.pooling_strategy = pooling_strategy

    def forward(self, batch: TokenizedBatch) -> TokenTaskLogits:
        subword_batch = contextualize_subwords(
            model=self.backbone,
            batch=batch,
        )
        token_batch = align_subwords_to_tokens(
            subword_batch=subword_batch,
            tokenized_batch=batch,
            pooling_strategy=self.pooling_strategy,
        )

        return self.heads(token_batch.hidden_states)


def build_pretrained_token_tagger(
    *,
    backbone_spec: PretrainedBackboneSpec,
    schema: TokenTaskSchema,
    dropout_probability: float,
    pooling_strategy: TokenPoolingStrategy = TokenPoolingStrategy.FIRST,
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
    )

    return TokenTagger(
        backbone=backbone,
        heads=heads,
        pooling_strategy=pooling_strategy,
    )
