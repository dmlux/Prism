from enum import StrEnum

from torch import Tensor, nn

from prism.modeling.outputs import TokenTaskLogits
from prism.schema import TokenTaskSchema


class TokenTaskHeadArchitecture(StrEnum):
    LINEAR = "linear"
    SHARED_MLP = "shared-mlp"
    WIDE_SHARED_MLP = "wide-shared-mlp"


class SharedResidualTokenProjection(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be greater than or equal to zero "
                "and less than one."
            )

        self.projection = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, hidden_states: Tensor) -> Tensor:
        projected_hidden_states = self.projection(hidden_states)
        activated_hidden_states = self.activation(projected_hidden_states)

        return hidden_states + self.dropout(activated_hidden_states)


class WideSharedResidualTokenProjection(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be greater than or equal to zero "
                "and less than one."
            )

        expanded_hidden_size = hidden_size * 2
        self.input_projection = nn.Linear(
            hidden_size,
            expanded_hidden_size,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)
        self.output_projection = nn.Linear(
            expanded_hidden_size,
            hidden_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        expanded_hidden_states = self.input_projection(hidden_states)
        activated_hidden_states = self.activation(expanded_hidden_states)
        projected_hidden_states = self.output_projection(
            self.dropout(activated_hidden_states)
        )

        return hidden_states + projected_hidden_states


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
        architecture: TokenTaskHeadArchitecture = TokenTaskHeadArchitecture.LINEAR,
    ) -> None:
        super().__init__()

        self.input_normalization = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.architecture = architecture
        self.input_projection: nn.Module
        if architecture is TokenTaskHeadArchitecture.LINEAR:
            self.input_projection = nn.Identity()
        elif architecture is TokenTaskHeadArchitecture.SHARED_MLP:
            self.input_projection = SharedResidualTokenProjection(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        elif architecture is TokenTaskHeadArchitecture.WIDE_SHARED_MLP:
            self.input_projection = WideSharedResidualTokenProjection(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        else:
            raise ValueError(f"Unsupported task-head architecture: {architecture!r}")

        self.upos_head = TokenClassificationHead(
            hidden_size=hidden_size,
            label_count=len(schema.upos.labels),
            dropout_probability=dropout_probability,
        )
        self.morphology_heads = nn.ModuleList(
            TokenClassificationHead(
                hidden_size=hidden_size,
                label_count=feature.logit_count,
                dropout_probability=dropout_probability,
            )
            for feature in schema.morphology.features
        )
        self.lemma_rule_head = TokenClassificationHead(
            hidden_size=hidden_size,
            label_count=len(schema.lemma_rules.rules),
            dropout_probability=dropout_probability,
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> TokenTaskLogits:
        normalized_hidden_states = self.input_normalization(hidden_states)
        projected_hidden_states = self.input_projection(normalized_hidden_states)

        return TokenTaskLogits(
            upos_logits=self.upos_head(projected_hidden_states),
            morphology_logits=tuple(
                head(projected_hidden_states) for head in self.morphology_heads
            ),
            lemma_rule_logits=self.lemma_rule_head(projected_hidden_states),
        )
