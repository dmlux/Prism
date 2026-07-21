import torch
from torch import Tensor, nn

from prism.schema import MorphologySchema


class StructuredMorphologyDecoder(nn.Module):
    """Refine morphology logits from soft cross-task and cross-feature context."""

    def __init__(
        self,
        *,
        hidden_size: int,
        upos_label_count: int,
        morphology_schema: MorphologySchema,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if upos_label_count <= 1:
            raise ValueError("UPOS label count must be greater than one.")
        if not morphology_schema.features:
            raise ValueError("Morphology schema must contain features.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be greater than or equal to zero "
                "and less than one."
            )

        context_size = upos_label_count + sum(
            feature.logit_count for feature in morphology_schema.features
        )
        self.upos_label_count = upos_label_count
        self.morphology_schema = morphology_schema
        self.context_normalization = nn.LayerNorm(
            context_size,
            elementwise_affine=False,
        )
        self.context_projection = nn.Linear(
            context_size,
            hidden_size,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)
        self.refinement_heads = nn.ModuleList(
            nn.Linear(hidden_size, feature.logit_count)
            for feature in morphology_schema.features
        )

        for refinement_head in self.refinement_heads:
            nn.init.zeros_(refinement_head.weight)
            nn.init.zeros_(refinement_head.bias)

    def forward(
        self,
        *,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        if upos_logits.ndim != 3 or upos_logits.shape[-1] != self.upos_label_count:
            raise ValueError("UPOS logits must match the decoder label count.")
        if len(morphology_logits) != len(self.morphology_schema.features):
            raise ValueError(
                "Morphology logits must match the structured decoder schema."
            )
        if any(
            feature_logits.ndim != 3
            or feature_logits.shape[:2] != upos_logits.shape[:2]
            or feature_logits.shape[-1] != feature.logit_count
            for feature_logits, feature in zip(
                morphology_logits,
                self.morphology_schema.features,
                strict=True,
            )
        ):
            raise ValueError(
                "Morphology logits must match token dimensions and feature schema."
            )

        context_parts = [torch.softmax(upos_logits, dim=-1)]
        context_parts.extend(
            torch.sigmoid(feature_logits)
            if feature.allows_multiple_values
            else torch.softmax(feature_logits, dim=-1)
            for feature_logits, feature in zip(
                morphology_logits,
                self.morphology_schema.features,
                strict=True,
            )
        )
        decision_context = torch.cat(context_parts, dim=-1)
        normalized_context = self.context_normalization(decision_context)
        contextual_refinement = self.dropout(
            self.activation(self.context_projection(normalized_context))
        )

        return tuple(
            feature_logits + refinement_head(contextual_refinement)
            for feature_logits, refinement_head in zip(
                morphology_logits,
                self.refinement_heads,
                strict=True,
            )
        )
