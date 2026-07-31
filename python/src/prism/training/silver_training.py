"""Soft-target knowledge-distillation loss and step for silver batches.

Silver sentences have no gold annotations. The student learns from the
calibrated teacher distributions stored in the label artifact: KL-style
cross-entropy against the complete UPOS and morphology distributions, and
against the renormalized lemma top-k distribution. Masked tokens (teacher
disagreement or low confidence) contribute nothing.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from prism.modeling.outputs import TokenTaskLogits
from prism.schema import MorphologySchema
from prism.training.silver_batches import SilverTokenTaskBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverKdLosses:
    upos_loss: Tensor
    morphology_loss: Tensor
    lemma_rule_loss: Tensor
    upos_token_count: int
    morphology_token_count: int
    lemma_token_count: int

    @property
    def total_loss(self) -> Tensor:
        return self.upos_loss + self.morphology_loss + self.lemma_rule_loss


def _masked_soft_cross_entropy(
    *,
    logits: Tensor,
    target_probabilities: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int]:
    token_count = int(mask.sum().item())
    if token_count == 0:
        return logits.sum() * 0.0, 0
    log_probabilities = functional.log_softmax(logits[mask], dim=-1)
    loss = -(target_probabilities[mask] * log_probabilities).sum(dim=-1).mean()
    return loss, token_count


def compute_silver_kd_loss(
    *,
    logits: TokenTaskLogits,
    batch: SilverTokenTaskBatch,
    morphology_schema: MorphologySchema,
) -> SilverKdLosses:
    token_mask = batch.model_inputs.token_mask
    upos_mask = batch.upos_mask & token_mask
    morphology_mask = batch.morphology_mask & token_mask
    lemma_mask = batch.lemma_mask & token_mask

    upos_loss, upos_token_count = _masked_soft_cross_entropy(
        logits=logits.upos_logits,
        target_probabilities=batch.upos_probabilities,
        mask=upos_mask,
    )

    morphology_token_count = int(morphology_mask.sum().item())
    if morphology_token_count == 0:
        morphology_loss = logits.upos_logits.sum() * 0.0
    else:
        feature_losses = []
        for feature, feature_logits, target_probabilities in zip(
            morphology_schema.features,
            logits.morphology_logits,
            batch.morphology_probabilities,
            strict=True,
        ):
            masked_logits = feature_logits[morphology_mask]
            masked_targets = target_probabilities[morphology_mask]
            if feature.allows_multiple_values:
                feature_losses.append(
                    functional.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_targets,
                        reduction="none",
                    )
                    .mean(dim=-1)
                    .mean()
                )
            else:
                log_probabilities = functional.log_softmax(masked_logits, dim=-1)
                feature_losses.append(
                    -(masked_targets * log_probabilities).sum(dim=-1).mean()
                )
        morphology_loss = torch.stack(feature_losses).mean()

    lemma_token_count = int(lemma_mask.sum().item())
    if lemma_token_count == 0:
        lemma_rule_loss = logits.lemma_rule_logits.sum() * 0.0
    else:
        student_log_probabilities = functional.log_softmax(
            logits.lemma_rule_logits[lemma_mask],
            dim=-1,
        )
        top_ids = batch.lemma_rule_ids[lemma_mask]
        top_probabilities = batch.lemma_rule_probabilities[lemma_mask]
        # The stored top-k covers ~99.3% of the teacher mass; renormalizing
        # makes it a proper distribution over the retained candidates.
        normalized = top_probabilities / top_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        gathered = student_log_probabilities.gather(-1, top_ids)
        lemma_rule_loss = -(normalized * gathered).sum(dim=-1).mean()

    return SilverKdLosses(
        upos_loss=upos_loss,
        morphology_loss=morphology_loss,
        lemma_rule_loss=lemma_rule_loss,
        upos_token_count=upos_token_count,
        morphology_token_count=morphology_token_count,
        lemma_token_count=lemma_token_count,
    )


def _forward_silver_model(
    *,
    model: nn.Module,
    batch: SilverTokenTaskBatch,
) -> TokenTaskLogits:
    if getattr(model, "character_encoder", None) is None:
        logits = model(batch.model_inputs)
    else:
        if batch.character_inputs is None:
            raise ValueError("Character-aware model requires character inputs.")
        logits = model(batch.model_inputs, batch.character_inputs)
    if not isinstance(logits, TokenTaskLogits):
        raise TypeError("Token-task model must return TokenTaskLogits.")
    return logits


def train_silver_kd_step(
    *,
    model: nn.Module,
    batch: SilverTokenTaskBatch,
    optimizer: torch.optim.Optimizer,
    max_gradient_norm: float,
    morphology_schema: MorphologySchema,
    silver_loss_weight: float,
) -> SilverKdLosses:
    if not math.isfinite(silver_loss_weight) or silver_loss_weight <= 0.0:
        raise ValueError("Silver loss weight must be finite and positive.")

    model.train()
    optimizer.zero_grad()

    logits = _forward_silver_model(model=model, batch=batch)
    losses = compute_silver_kd_loss(
        logits=logits,
        batch=batch,
        morphology_schema=morphology_schema,
    )
    (silver_loss_weight * losses.total_loss).backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
    optimizer.step()

    return losses
