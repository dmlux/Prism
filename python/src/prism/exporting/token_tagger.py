from torch import Tensor, nn

from prism.modeling import (
    CharacterTokenBatch,
    MorphologyLogitCorrection,
    TokenizedBatch,
)


class MorphologyLogitCorrectionExportLayer(nn.Module):
    """Store fixed morphology corrections inside the exported model graph."""

    def __init__(self, *, correction: MorphologyLogitCorrection) -> None:
        super().__init__()
        self.feature_count = len(correction.weights)

        for index, weights in enumerate(correction.weights):
            self.register_buffer(
                f"offset_{index}",
                correction.strength * weights.detach().clone().log(),
            )

    def forward(
        self,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        if len(morphology_logits) != self.feature_count:
            raise ValueError(
                "Morphology correction must match exported feature logits."
            )

        return tuple(
            feature_logits - getattr(self, f"offset_{index}")
            for index, feature_logits in enumerate(morphology_logits)
        )


class TokenTaggerExportAdapter(nn.Module):
    """Expose the token tagger through a flat tensor-only export contract."""

    def __init__(
        self,
        *,
        model: nn.Module,
        morphology_logit_correction: MorphologyLogitCorrection | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.morphology_logit_correction = (
            None
            if morphology_logit_correction is None
            else MorphologyLogitCorrectionExportLayer(
                correction=morphology_logit_correction
            )
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        first_subword_indices: Tensor,
        subword_end_indices: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        output = self.model(
            TokenizedBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_indices=first_subword_indices,
                subword_end_indices=subword_end_indices,
                token_mask=token_mask,
            )
        )

        morphology_logits = output.morphology_logits
        if self.morphology_logit_correction is not None:
            morphology_logits = self.morphology_logit_correction(morphology_logits)

        return (
            output.upos_logits,
            *morphology_logits,
            output.lemma_rule_logits,
        )


class CharacterAwareTokenTaggerExportAdapter(nn.Module):
    """Expose a character-aware token tagger through flat tensor inputs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        morphology_logit_correction: MorphologyLogitCorrection | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.morphology_logit_correction = (
            None
            if morphology_logit_correction is None
            else MorphologyLogitCorrectionExportLayer(
                correction=morphology_logit_correction
            )
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        first_subword_indices: Tensor,
        subword_end_indices: Tensor,
        token_mask: Tensor,
        character_ids: Tensor,
        character_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        output = self.model(
            TokenizedBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_indices=first_subword_indices,
                subword_end_indices=subword_end_indices,
                token_mask=token_mask,
            ),
            CharacterTokenBatch(
                character_ids=character_ids,
                character_mask=character_mask,
                token_mask=token_mask,
            ),
        )

        morphology_logits = output.morphology_logits
        if self.morphology_logit_correction is not None:
            morphology_logits = self.morphology_logit_correction(morphology_logits)

        return (
            output.upos_logits,
            *morphology_logits,
            output.lemma_rule_logits,
        )
