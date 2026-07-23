"""Local sentence-level agreement refinement for morphology logits."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.schema import MorphologySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyAgreementRefinerSpec:
    window_radius: int
    bottleneck_size: int
    target_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.window_radius <= 0:
            raise ValueError("Agreement window radius must be positive.")
        if self.bottleneck_size <= 0:
            raise ValueError("Agreement bottleneck size must be positive.")
        if not self.target_feature_names:
            raise ValueError("Agreement refiner requires target features.")
        if any(not name or name.strip() != name for name in self.target_feature_names):
            raise ValueError(
                "Agreement target feature names must be non-empty and trimmed."
            )
        if len(set(self.target_feature_names)) != len(self.target_feature_names):
            raise ValueError("Agreement target feature names must be unique.")


def validate_morphology_agreement_refiner_spec(
    *,
    spec: MorphologyAgreementRefinerSpec,
    morphology_schema: MorphologySchema,
) -> None:
    feature_names = {feature.name for feature in morphology_schema.features}
    unknown_names = sorted(set(spec.target_feature_names).difference(feature_names))
    if unknown_names:
        raise ValueError(f"Unknown agreement target feature: {unknown_names[0]!r}")


class MorphologyAgreementRefiner(nn.Module):
    """Add gated corrections using nearby tokens' soft task evidence.

    The local attention excludes the current token. Each token therefore sees
    only neighboring model evidence inside the configured sentence window.
    The existing independent, structured, and bundle-aware logits remain the
    residual fallback.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        upos_label_count: int,
        morphology_schema: MorphologySchema,
        spec: MorphologyAgreementRefinerSpec,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if upos_label_count <= 1:
            raise ValueError("UPOS label count must be greater than one.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Dropout probability must be in [0,1).")
        validate_morphology_agreement_refiner_spec(
            spec=spec,
            morphology_schema=morphology_schema,
        )

        self.morphology_schema = morphology_schema
        self.spec = spec
        self.enabled = True
        evidence_size = sum(
            feature.logit_count for feature in morphology_schema.features
        )

        self.hidden_projection = nn.Linear(hidden_size, spec.bottleneck_size)
        self.upos_projection = nn.Linear(
            upos_label_count,
            spec.bottleneck_size,
            bias=False,
        )
        self.morphology_projection = nn.Linear(
            evidence_size,
            spec.bottleneck_size,
            bias=False,
        )
        self.input_normalization = nn.LayerNorm(spec.bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)
        self.query_projection = nn.Linear(
            spec.bottleneck_size,
            spec.bottleneck_size,
            bias=False,
        )
        self.key_projection = nn.Linear(
            spec.bottleneck_size,
            spec.bottleneck_size,
            bias=False,
        )
        self.value_projection = nn.Linear(
            spec.bottleneck_size,
            spec.bottleneck_size,
            bias=False,
        )

        feature_indices = {
            feature.name: index
            for index, feature in enumerate(morphology_schema.features)
        }
        self.target_feature_indices = tuple(
            feature_indices[name] for name in spec.target_feature_names
        )
        self.correction_heads = nn.ModuleList(
            nn.Linear(
                spec.bottleneck_size,
                morphology_schema.features[index].logit_count,
            )
            for index in self.target_feature_indices
        )
        self.refinement_gates = nn.Parameter(
            torch.full((len(self.target_feature_indices),), -2.0)
        )
        for head in self.correction_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def forward(
        self,
        *,
        hidden_states: Tensor,
        token_mask: Tensor,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        if not self.enabled:
            return morphology_logits
        if hidden_states.ndim != 3:
            raise ValueError("Agreement hidden states must have three dimensions.")
        if (
            token_mask.shape != hidden_states.shape[:2]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("Agreement token mask must match hidden states.")
        if upos_logits.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("Agreement UPOS logits must match token dimensions.")
        if len(morphology_logits) != len(self.morphology_schema.features):
            raise ValueError("Agreement morphology logits must match the schema.")

        morphology_probabilities = tuple(
            torch.sigmoid(feature_logits)
            if feature.allows_multiple_values
            else torch.softmax(feature_logits, dim=-1)
            for feature_logits, feature in zip(
                morphology_logits,
                self.morphology_schema.features,
                strict=True,
            )
        )
        evidence = (
            self.hidden_projection(hidden_states)
            + self.upos_projection(torch.softmax(upos_logits, dim=-1))
            + self.morphology_projection(torch.cat(morphology_probabilities, dim=-1))
        )
        evidence = self.dropout(self.activation(self.input_normalization(evidence)))

        query = self.query_projection(evidence)
        key = self.key_projection(evidence)
        value = self.value_projection(evidence)
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / (self.spec.bottleneck_size**0.5)

        token_count = hidden_states.shape[1]
        positions = torch.arange(token_count, device=hidden_states.device)
        distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        local_neighbor_mask = (
            (distances <= self.spec.window_radius) & (distances > 0)
        ).unsqueeze(0)
        attention_mask = local_neighbor_mask & token_mask.unsqueeze(1)
        attention_weights = torch.softmax(
            attention_scores.masked_fill(
                ~attention_mask,
                torch.finfo(attention_scores.dtype).min,
            ),
            dim=-1,
        )
        attention_weights = attention_weights * attention_mask.to(
            dtype=attention_weights.dtype
        )
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(attention_weights.dtype).eps)
        context = attention_weights @ value
        context = self.dropout(self.activation(context))

        refined_logits = list(morphology_logits)
        valid_tokens = token_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        has_neighbor = attention_mask.any(dim=-1, keepdim=True).to(
            dtype=hidden_states.dtype
        )
        for target_index, (feature_index, head) in enumerate(
            zip(
                self.target_feature_indices,
                self.correction_heads,
                strict=True,
            )
        ):
            gate = torch.sigmoid(self.refinement_gates[target_index])
            correction = head(context) * valid_tokens * has_neighbor
            refined_logits[feature_index] = (
                refined_logits[feature_index] + gate * correction
            )

        return tuple(refined_logits)
