from enum import StrEnum

from torch import Tensor, nn

from prism.modeling.character_encoders import CharacterResidualFusion
from prism.modeling.outputs import TokenTaskLogits
from prism.modeling.morphology_bundle_reranker import (
    MorphologyBundleReranker,
    MorphologyBundleRerankerSpec,
)
from prism.modeling.morphology_agreement import (
    MorphologyAgreementRefiner,
    MorphologyAgreementRefinerSpec,
)
from prism.modeling.structured_morphology import StructuredMorphologyDecoder
from prism.schema import TokenTaskSchema


class TokenTaskHeadArchitecture(StrEnum):
    LINEAR = "linear"
    SHARED_MLP = "shared-mlp"
    WIDE_SHARED_MLP = "wide-shared-mlp"
    WIDE_SHARED_MLP_TASK_ADAPTERS = "wide-shared-mlp-task-adapters"
    WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY = "wide-shared-mlp-structured-morphology"
    WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN = (
        "wide-shared-mlp-structured-morphology-character-cnn"
    )

    @property
    def uses_character_encoder(self) -> bool:
        return (
            self
            is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
        )


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


class TaskResidualAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck_size: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if bottleneck_size <= 0 or bottleneck_size >= hidden_size:
            raise ValueError(
                "Adapter bottleneck size must be positive and smaller than "
                "the hidden size."
            )
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be greater than or equal to zero "
                "and less than one."
            )

        self.input_projection = nn.Linear(
            hidden_size,
            bottleneck_size,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)
        self.output_projection = nn.Linear(
            bottleneck_size,
            hidden_size,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        bottleneck_hidden_states = self.input_projection(hidden_states)
        activated_hidden_states = self.activation(bottleneck_hidden_states)
        adapted_hidden_states = self.output_projection(
            self.dropout(activated_hidden_states)
        )

        return hidden_states + adapted_hidden_states


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
        morphology_bundle_reranker_spec: MorphologyBundleRerankerSpec | None = None,
        morphology_agreement_refiner_spec: (
            MorphologyAgreementRefinerSpec | None
        ) = None,
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
        elif architecture in (
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP,
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_TASK_ADAPTERS,
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY,
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN,
        ):
            self.input_projection = WideSharedResidualTokenProjection(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        else:
            raise ValueError(f"Unsupported task-head architecture: {architecture!r}")

        self.upos_adapter: nn.Module
        self.morphology_adapter: nn.Module
        self.lemma_adapter: nn.Module
        if architecture is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_TASK_ADAPTERS:
            bottleneck_size = max(1, hidden_size // 2)
            self.upos_adapter = TaskResidualAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                dropout_probability=dropout_probability,
            )
            self.morphology_adapter = TaskResidualAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                dropout_probability=dropout_probability,
            )
            self.lemma_adapter = TaskResidualAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
                dropout_probability=dropout_probability,
            )
        else:
            self.upos_adapter = nn.Identity()
            self.morphology_adapter = nn.Identity()
            self.lemma_adapter = nn.Identity()

        self.character_fusion: CharacterResidualFusion | None
        if architecture.uses_character_encoder:
            self.character_fusion = CharacterResidualFusion(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        else:
            self.character_fusion = None

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
        self.structured_morphology_decoder: StructuredMorphologyDecoder | None
        if architecture in (
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY,
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN,
        ):
            self.structured_morphology_decoder = StructuredMorphologyDecoder(
                hidden_size=hidden_size,
                upos_label_count=len(schema.upos.labels),
                morphology_schema=schema.morphology,
                dropout_probability=dropout_probability,
            )
        else:
            self.structured_morphology_decoder = None

        self.morphology_bundle_reranker: MorphologyBundleReranker | None
        if morphology_bundle_reranker_spec is None:
            self.morphology_bundle_reranker = None
        else:
            self.morphology_bundle_reranker = MorphologyBundleReranker(
                hidden_size=hidden_size,
                upos_label_count=len(schema.upos.labels),
                morphology_schema=schema.morphology,
                spec=morphology_bundle_reranker_spec,
                dropout_probability=dropout_probability,
            )

        self.morphology_agreement_refiner: MorphologyAgreementRefiner | None
        if morphology_agreement_refiner_spec is None:
            self.morphology_agreement_refiner = None
        else:
            self.morphology_agreement_refiner = MorphologyAgreementRefiner(
                hidden_size=hidden_size,
                upos_label_count=len(schema.upos.labels),
                morphology_schema=schema.morphology,
                spec=morphology_agreement_refiner_spec,
                dropout_probability=dropout_probability,
            )

    def set_morphology_bundle_loss_gradient_isolation(
        self,
        enabled: bool,
    ) -> None:
        if self.morphology_bundle_reranker is None:
            if enabled:
                raise ValueError(
                    "Bundle-loss gradient isolation requires a bundle reranker."
                )
            return
        self.morphology_bundle_reranker.set_direct_loss_gradient_isolation(enabled)

    def forward(
        self,
        hidden_states: Tensor,
        character_hidden_states: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> TokenTaskLogits:
        normalized_hidden_states = self.input_normalization(hidden_states)
        projected_hidden_states = self.input_projection(normalized_hidden_states)
        upos_hidden_states = self.upos_adapter(projected_hidden_states)
        task_hidden_states = projected_hidden_states
        if self.character_fusion is not None:
            if character_hidden_states is None:
                raise ValueError(
                    "Character-aware task heads require character hidden states."
                )
            task_hidden_states = self.character_fusion(
                contextual_hidden_states=projected_hidden_states,
                character_hidden_states=character_hidden_states,
            )
        elif character_hidden_states is not None:
            raise ValueError(
                "Character hidden states require a character-aware architecture."
            )

        morphology_hidden_states = self.morphology_adapter(task_hidden_states)
        lemma_hidden_states = self.lemma_adapter(task_hidden_states)
        upos_logits = self.upos_head(upos_hidden_states)
        morphology_logits = tuple(
            head(morphology_hidden_states) for head in self.morphology_heads
        )
        morphology_bundle_scores = None
        morphology_bundle_loss_scores = None
        if self.structured_morphology_decoder is not None:
            morphology_logits = self.structured_morphology_decoder(
                upos_logits=upos_logits,
                morphology_logits=morphology_logits,
            )
        if self.morphology_bundle_reranker is not None:
            (
                morphology_logits,
                morphology_bundle_scores,
                morphology_bundle_loss_scores,
            ) = self.morphology_bundle_reranker.refine_with_training_scores(
                hidden_states=morphology_hidden_states,
                upos_logits=upos_logits,
                morphology_logits=morphology_logits,
            )
        if self.morphology_agreement_refiner is not None:
            if token_mask is None:
                raise ValueError("Agreement-aware task heads require a token mask.")
            morphology_logits = self.morphology_agreement_refiner(
                hidden_states=morphology_hidden_states,
                token_mask=token_mask,
                upos_logits=upos_logits,
                morphology_logits=morphology_logits,
            )

        return TokenTaskLogits(
            upos_logits=upos_logits,
            morphology_logits=morphology_logits,
            lemma_rule_logits=self.lemma_rule_head(lemma_hidden_states),
            morphology_bundle_scores=morphology_bundle_scores,
            morphology_bundle_loss_scores=morphology_bundle_loss_scores,
        )
