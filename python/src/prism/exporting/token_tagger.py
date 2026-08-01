from collections.abc import Sequence

import torch
from torch import Tensor, nn

from prism.modeling import (
    CharacterTokenBatch,
    MorphologyLogitCorrection,
    TokenizedBatch,
)


class CalibratedProbabilityExportLayer(nn.Module):
    """Turn corrected logits into calibrated probabilities inside the graph.

    Mirrors ``calibrated_task_probabilities``: every head is scaled by its
    fitted temperature, exclusive heads go through softmax, multi-valued
    morphology features through independent sigmoids. Baking this into the
    export means native runtimes receive final, trustworthy probabilities and
    implement no decoding mathematics beyond argmax and thresholding.
    """

    def __init__(
        self,
        *,
        upos_temperature: float,
        morphology_temperatures: Sequence[float],
        multi_valued_features: Sequence[bool],
        lemma_rule_temperature: float,
    ) -> None:
        super().__init__()
        if len(morphology_temperatures) != len(multi_valued_features):
            raise ValueError(
                "Calibration temperatures must match the morphology features."
            )
        if any(
            temperature <= 0.0
            for temperature in (
                upos_temperature,
                lemma_rule_temperature,
                *morphology_temperatures,
            )
        ):
            raise ValueError("Calibration temperatures must be positive.")
        self.upos_temperature = float(upos_temperature)
        self.morphology_temperatures = tuple(
            float(temperature) for temperature in morphology_temperatures
        )
        self.multi_valued_features = tuple(bool(flag) for flag in multi_valued_features)
        self.lemma_rule_temperature = float(lemma_rule_temperature)

    def forward(
        self,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
        lemma_rule_logits: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        if len(morphology_logits) != len(self.morphology_temperatures):
            raise ValueError("Calibration must match the exported morphology features.")
        # Calibration always computes in float32: softmax over fp16 logits is
        # both the numerically riskiest spot and not universally covered by
        # portable kernels, and applications receive fp32 probabilities.
        morphology_probabilities = tuple(
            torch.sigmoid(feature_logits.float() / temperature)
            if multi_valued
            else torch.softmax(feature_logits.float() / temperature, dim=-1)
            for feature_logits, temperature, multi_valued in zip(
                morphology_logits,
                self.morphology_temperatures,
                self.multi_valued_features,
            )
        )
        return (
            torch.softmax(upos_logits.float() / self.upos_temperature, dim=-1),
            morphology_probabilities,
            torch.softmax(
                lemma_rule_logits.float() / self.lemma_rule_temperature,
                dim=-1,
            ),
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

        # The correction tail always computes in float32 so fp16-lowered
        # programs keep supported portable kernels and full output precision.
        return tuple(
            feature_logits.float() - getattr(self, f"offset_{index}").float()
            for index, feature_logits in enumerate(morphology_logits)
        )


class TokenTaggerExportAdapter(nn.Module):
    """Expose the token tagger through a flat tensor-only export contract."""

    def __init__(
        self,
        *,
        model: nn.Module,
        morphology_logit_correction: MorphologyLogitCorrection | None = None,
        calibrated_probabilities: CalibratedProbabilityExportLayer | None = None,
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
        self.calibrated_probabilities = calibrated_probabilities

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

        if self.calibrated_probabilities is not None:
            upos, morphology, lemma_rule = self.calibrated_probabilities(
                output.upos_logits,
                morphology_logits,
                output.lemma_rule_logits,
            )
            return (upos, *morphology, lemma_rule)

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
        calibrated_probabilities: CalibratedProbabilityExportLayer | None = None,
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
        self.calibrated_probabilities = calibrated_probabilities

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

        if self.calibrated_probabilities is not None:
            upos, morphology, lemma_rule = self.calibrated_probabilities(
                output.upos_logits,
                morphology_logits,
                output.lemma_rule_logits,
            )
            return (upos, *morphology, lemma_rule)

        return (
            output.upos_logits,
            *morphology_logits,
            output.lemma_rule_logits,
        )
