from torch import nn

from prism.modeling.alignment import align_subwords_to_tokens
from prism.modeling.batches import TokenizedBatch
from prism.modeling.encoders import contextualize_subwords
from prism.modeling.heads import TokenTaskHeads
from prism.modeling.outputs import TokenTaskLogits


class TokenTagger(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        heads: TokenTaskHeads,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.heads = heads

    def forward(self, batch: TokenizedBatch) -> TokenTaskLogits:
        subword_batch = contextualize_subwords(
            model=self.backbone,
            batch=batch,
        )
        token_batch = align_subwords_to_tokens(
            subword_batch=subword_batch,
            tokenized_batch=batch,
        )

        return self.heads(token_batch.hidden_states)
