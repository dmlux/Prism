from enum import StrEnum

import torch
from torch import Tensor, nn

from prism.modeling.character_encoders import CharacterResidualFusion
from prism.modeling.outputs import TokenTaskHiddenStates, TokenTaskLogits
from prism.modeling.morphology_bundle_reranker import (
    MorphologyBundleLossGradientScope,
    MorphologyBundleLossInputs,
    MorphologyBundleReranker,
    MorphologyBundleRerankerSpec,
)
from prism.modeling.structured_morphology import StructuredMorphologyDecoder
from prism.schema import TokenTaskSchema


class TokenTaskHeadArchitecture(StrEnum):
    LINEAR = "linear"
    WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN = (
        "wide-shared-mlp-structured-morphology-character-cnn"
    )

    @property
    def uses_character_encoder(self) -> bool:
        return (
            self
            is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
        )


class MorphologyPreHeadArchitecture(StrEnum):
    IDENTITY = "identity"
    SHARED_MLP = "shared-mlp"


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
        morphology_pre_head_architecture: MorphologyPreHeadArchitecture = (
            MorphologyPreHeadArchitecture.IDENTITY
        ),
        morphology_bundle_reranker_spec: MorphologyBundleRerankerSpec | None = None,
    ) -> None:
        super().__init__()

        self.input_normalization = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.architecture = architecture
        self.morphology_pre_head_architecture = morphology_pre_head_architecture
        self.input_projection: nn.Module
        if architecture is TokenTaskHeadArchitecture.LINEAR:
            self.input_projection = nn.Identity()
        elif (
            architecture
            is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
        ):
            self.input_projection = WideSharedResidualTokenProjection(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        else:
            raise ValueError(f"Unsupported task-head architecture: {architecture!r}")

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
        if (
            architecture
            is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
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

        self.morphology_pre_head_projection: nn.Module
        if morphology_pre_head_architecture is MorphologyPreHeadArchitecture.IDENTITY:
            self.morphology_pre_head_projection = nn.Identity()
        elif (
            morphology_pre_head_architecture is MorphologyPreHeadArchitecture.SHARED_MLP
        ):
            self.morphology_pre_head_projection = WideSharedResidualTokenProjection(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        else:
            raise ValueError(
                "Unsupported morphology pre-head architecture: "
                f"{morphology_pre_head_architecture!r}"
            )

    def set_morphology_bundle_loss_gradient_scope(
        self,
        scope: MorphologyBundleLossGradientScope,
    ) -> None:
        if self.morphology_bundle_reranker is None:
            if scope is not MorphologyBundleLossGradientScope.FULL:
                raise ValueError(
                    "A restricted bundle-loss gradient requires a bundle reranker."
                )
            return
        self.morphology_bundle_reranker.set_direct_loss_gradient_scope(scope)

    def _calculate_morphology_logits(
        self,
        *,
        morphology_hidden_states: Tensor,
        upos_logits: Tensor,
    ) -> tuple[Tensor, ...]:
        morphology_logits = tuple(
            head(morphology_hidden_states) for head in self.morphology_heads
        )
        if self.structured_morphology_decoder is not None:
            morphology_logits = self.structured_morphology_decoder(
                upos_logits=upos_logits,
                morphology_logits=morphology_logits,
            )
        return morphology_logits

    @staticmethod
    def _preserve_value_with_gradient_from(
        *,
        value: Tensor,
        gradient_source: Tensor,
    ) -> Tensor:
        return gradient_source + (value - gradient_source).detach()

    def _morphology_bundle_loss_inputs(
        self,
        *,
        task_hidden_states: Tensor,
        morphology_hidden_states: Tensor,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> MorphologyBundleLossInputs | None:
        reranker = self.morphology_bundle_reranker
        if (
            reranker is None
            or reranker.direct_loss_gradient_scope
            is not MorphologyBundleLossGradientScope.MORPHOLOGY
            or not torch.is_grad_enabled()
        ):
            return None

        protected_morphology_hidden_states = self._encode_morphology_hidden_states(
            task_hidden_states.detach()
        )
        protected_morphology_logits = self._calculate_morphology_logits(
            morphology_hidden_states=protected_morphology_hidden_states,
            upos_logits=upos_logits.detach(),
        )
        return MorphologyBundleLossInputs(
            hidden_states=self._preserve_value_with_gradient_from(
                value=morphology_hidden_states,
                gradient_source=protected_morphology_hidden_states,
            ),
            upos_logits=upos_logits.detach(),
            morphology_logits=tuple(
                self._preserve_value_with_gradient_from(
                    value=value,
                    gradient_source=gradient_source,
                )
                for value, gradient_source in zip(
                    morphology_logits,
                    protected_morphology_logits,
                    strict=True,
                )
            ),
        )

    def _encode_morphology_hidden_states(
        self,
        task_hidden_states: Tensor,
    ) -> Tensor:
        return self.morphology_pre_head_projection(task_hidden_states)

    def encode_hidden_states(
        self,
        hidden_states: Tensor,
        character_hidden_states: Tensor | None = None,
    ) -> TokenTaskHiddenStates:
        normalized_hidden_states = self.input_normalization(hidden_states)
        projected_hidden_states = self.input_projection(normalized_hidden_states)
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

        morphology_hidden_states = self._encode_morphology_hidden_states(
            task_hidden_states
        )
        return TokenTaskHiddenStates(
            task=task_hidden_states,
            upos=projected_hidden_states,
            morphology=morphology_hidden_states,
            lemma=task_hidden_states,
        )

    def classify_hidden_states(
        self,
        hidden_states: TokenTaskHiddenStates,
    ) -> TokenTaskLogits:
        upos_logits = self.upos_head(hidden_states.upos)
        morphology_logits = self._calculate_morphology_logits(
            morphology_hidden_states=hidden_states.morphology,
            upos_logits=upos_logits,
        )
        morphology_bundle_scores = None
        morphology_bundle_loss_scores = None
        if self.morphology_bundle_reranker is not None:
            morphology_bundle_loss_inputs = self._morphology_bundle_loss_inputs(
                task_hidden_states=hidden_states.task,
                morphology_hidden_states=hidden_states.morphology,
                upos_logits=upos_logits,
                morphology_logits=morphology_logits,
            )
            (
                morphology_logits,
                morphology_bundle_scores,
                morphology_bundle_loss_scores,
            ) = self.morphology_bundle_reranker.refine_with_training_scores(
                hidden_states=hidden_states.morphology,
                upos_logits=upos_logits,
                morphology_logits=morphology_logits,
                morphology_loss_inputs=morphology_bundle_loss_inputs,
            )
        return TokenTaskLogits(
            upos_logits=upos_logits,
            morphology_logits=morphology_logits,
            lemma_rule_logits=self.lemma_rule_head(hidden_states.lemma),
            morphology_bundle_scores=morphology_bundle_scores,
            morphology_bundle_loss_scores=morphology_bundle_loss_scores,
        )

    def forward(
        self,
        hidden_states: Tensor,
        character_hidden_states: Tensor | None = None,
    ) -> TokenTaskLogits:
        task_hidden_states = self.encode_hidden_states(
            hidden_states,
            character_hidden_states=character_hidden_states,
        )
        return self.classify_hidden_states(task_hidden_states)
