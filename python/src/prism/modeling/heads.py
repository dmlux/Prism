from torch import Tensor, nn

from prism.modeling.outputs import TokenTaskLogits
from prism.schema import TokenTaskSchema


class TokenClassificationHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        label_count: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if label_count <= 1:
            raise ValueError("Label count must be greater than one.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be greater than or equal to zero "
                "and less than one."
            )

        self.dropout = nn.Dropout(dropout_probability)
        self.projection = nn.Linear(hidden_size, label_count)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(self.dropout(hidden_states))


class TokenTaskHeads(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        schema: TokenTaskSchema,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        self.upos_head = TokenClassificationHead(
            hidden_size=hidden_size,
            label_count=len(schema.upos.labels),
            dropout_probability=dropout_probability,
        )
        self.morphology_heads = nn.ModuleList(
            TokenClassificationHead(
                hidden_size=hidden_size,
                label_count=len(feature.labels),
                dropout_probability=dropout_probability,
            )
            for feature in schema.morphology.features
        )
        self.lemma_rule_head = TokenClassificationHead(
            hidden_size=hidden_size,
            label_count=len(schema.lemma_rules.rules),
            dropout_probability=dropout_probability,
        )

    def forward(self, hidden_states: Tensor) -> TokenTaskLogits:
        return TokenTaskLogits(
            upos_logits=self.upos_head(hidden_states),
            morphology_logits=tuple(
                head(hidden_states) for head in self.morphology_heads
            ),
            lemma_rule_logits=self.lemma_rule_head(hidden_states),
        )
