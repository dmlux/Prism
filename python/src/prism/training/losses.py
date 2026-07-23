from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional

from prism.data import TokenTaskTargetBatch
from prism.modeling import MorphologyBundleRerankerSpec, TokenTaskLogits
from prism.schema import MorphologySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskLossWeights:
    morphology_weights: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.morphology_weights:
            raise ValueError("Loss weights must contain morphology features.")

        for weights in self.morphology_weights:
            if weights.ndim != 1:
                raise ValueError("Morphology weights must have one dimension.")
            if not weights.is_floating_point():
                raise ValueError("Morphology weights must be floating point.")
            if not torch.isfinite(weights).all():
                raise ValueError("Morphology weights must be finite.")
            if torch.any(weights <= 0):
                raise ValueError("Morphology weights must be positive.")

    def to(
        self,
        device: torch.device,
    ) -> "TokenTaskLossWeights":
        return TokenTaskLossWeights(
            morphology_weights=tuple(
                weights.to(device=device) for weights in self.morphology_weights
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskLosses:
    upos_loss: Tensor
    morphology_loss: Tensor
    lemma_rule_loss: Tensor
    total_loss: Tensor
    morphology_bundle_loss: Tensor | None = None
    morphology_bundle_target_count: int = 0
    morphology_bundle_token_count: int = 0
    morphology_bundle_loss_weight: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleLossPolicy:
    weight: float
    candidate_morphology_targets: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(
                "Morphology bundle loss weight must be finite and non-negative."
            )
        if not self.candidate_morphology_targets:
            raise ValueError(
                "Morphology bundle loss requires candidate morphology targets."
            )
        candidate_count = self.candidate_morphology_targets[0].shape[0]
        if candidate_count <= 0:
            raise ValueError("Morphology bundle loss requires candidates.")
        for targets in self.candidate_morphology_targets:
            if targets.ndim != 2 or targets.shape[0] != candidate_count:
                raise ValueError(
                    "Bundle candidate targets must share one candidate dimension."
                )
            if targets.dtype != torch.bool:
                raise ValueError("Bundle candidate targets must use torch.bool.")

    @classmethod
    def from_reranker_spec(
        cls,
        *,
        spec: MorphologyBundleRerankerSpec,
        weight: float,
    ) -> "MorphologyBundleLossPolicy":
        feature_count = len(spec.candidates[0].morphology)
        return cls(
            weight=weight,
            candidate_morphology_targets=tuple(
                torch.tensor(
                    [
                        candidate.morphology[feature_index]
                        for candidate in spec.candidates
                    ],
                    dtype=torch.bool,
                )
                for feature_index in range(feature_count)
            ),
        )

    def to(self, device: torch.device) -> "MorphologyBundleLossPolicy":
        return MorphologyBundleLossPolicy(
            weight=self.weight,
            candidate_morphology_targets=tuple(
                targets.to(device=device)
                for targets in self.candidate_morphology_targets
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleLossResult:
    loss: Tensor
    target_count: int
    token_count: int

    @property
    def coverage(self) -> float:
        return 0.0 if self.token_count == 0 else self.target_count / self.token_count


def _masked_mean(
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    selected_values = values.masked_select(mask)

    if selected_values.numel() == 0:
        return values.sum() * 0.0

    return selected_values.mean()


def calculate_morphology_bundle_loss(
    *,
    candidate_scores: Tensor,
    morphology_targets: tuple[Tensor, ...],
    token_mask: Tensor,
    policy: MorphologyBundleLossPolicy,
) -> MorphologyBundleLossResult:
    if candidate_scores.ndim != 3:
        raise ValueError("Morphology bundle scores must have three dimensions.")
    if candidate_scores.shape[:2] != token_mask.shape:
        raise ValueError("Bundle scores and token mask must share token dimensions.")
    if token_mask.dtype != torch.bool:
        raise ValueError("Morphology bundle token mask must use torch.bool.")
    if len(morphology_targets) != len(policy.candidate_morphology_targets):
        raise ValueError(
            "Morphology bundle targets must match the candidate feature count."
        )
    if candidate_scores.shape[-1] != (policy.candidate_morphology_targets[0].shape[0]):
        raise ValueError("Morphology bundle scores must match candidate count.")

    matching_candidates = torch.ones(
        candidate_scores.shape,
        dtype=torch.bool,
        device=candidate_scores.device,
    )
    for targets, candidate_targets in zip(
        morphology_targets,
        policy.candidate_morphology_targets,
        strict=True,
    ):
        if targets.shape[:2] != token_mask.shape:
            raise ValueError("Morphology bundle targets must share token dimensions.")
        if targets.shape[-1] != candidate_targets.shape[-1]:
            raise ValueError(
                "Morphology bundle targets must match candidate label dimensions."
            )
        matching_candidates &= (
            targets.unsqueeze(-2) == candidate_targets.unsqueeze(0).unsqueeze(0)
        ).all(dim=-1)

    covered_tokens = token_mask & matching_candidates.any(dim=-1)
    token_count = int(token_mask.sum().item())
    target_count = int(covered_tokens.sum().item())
    log_probabilities = functional.log_softmax(candidate_scores, dim=-1)
    gold_log_probability = torch.logsumexp(
        log_probabilities.masked_fill(~matching_candidates, -torch.inf),
        dim=-1,
    )
    per_token_loss = torch.where(
        covered_tokens,
        -gold_log_probability,
        torch.zeros_like(gold_log_probability),
    )

    return MorphologyBundleLossResult(
        loss=_masked_mean(per_token_loss, covered_tokens),
        target_count=target_count,
        token_count=token_count,
    )


def compute_token_task_loss(
    *,
    logits: TokenTaskLogits,
    targets: TokenTaskTargetBatch,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
) -> TokenTaskLosses:
    if logits.upos_logits.shape[:2] != targets.upos_ids.shape:
        raise ValueError("Logits and targets must share batch and token dimensions.")
    if logits.morphology_feature_count != targets.morphology_feature_count:
        raise ValueError(
            "Morphology logits and targets must contain the same features."
        )
    if logits.morphology_feature_count != len(morphology_schema.features):
        raise ValueError("Morphology logits must match the morphology schema.")

    upos_per_token = functional.cross_entropy(
        logits.upos_logits.transpose(1, 2),
        targets.upos_ids,
        reduction="none",
    )
    upos_loss = _masked_mean(
        upos_per_token,
        targets.token_mask,
    )

    morphology_feature_losses: list[Tensor] = []

    if loss_weights is None:
        morphology_weights: tuple[Tensor | None, ...] = (
            None,
        ) * logits.morphology_feature_count
    else:
        if len(loss_weights.morphology_weights) != logits.morphology_feature_count:
            raise ValueError("Morphology loss weights must match the feature count.")

        morphology_weights = loss_weights.morphology_weights

    for feature_logits, feature_targets, feature_schema, feature_weights in zip(
        logits.morphology_logits,
        targets.morphology_targets,
        morphology_schema.features,
        morphology_weights,
        strict=True,
    ):
        expected_target_shape = (*feature_logits.shape[:2], len(feature_schema.labels))
        if feature_targets.shape != expected_target_shape:
            raise ValueError(
                "Morphology targets must match the feature's complete label space."
            )
        if (
            feature_weights is not None
            and feature_weights.shape[0] != feature_schema.logit_count
        ):
            raise ValueError("Morphology weights must match the feature logit count.")

        valid_targets = feature_targets[targets.token_mask]

        if feature_schema.allows_multiple_values:
            none_targets = valid_targets[..., 0]
            value_targets = valid_targets[..., 1:]
            if torch.any(none_targets != ~value_targets.any(dim=-1)):
                raise ValueError(
                    "Multi-label morphology targets must derive <NONE> from values."
                )

            feature_per_label = functional.binary_cross_entropy_with_logits(
                feature_logits,
                feature_targets[..., 1:].to(feature_logits.dtype),
                pos_weight=feature_weights,
                reduction="none",
            )
            feature_per_token = feature_per_label.mean(dim=-1)
        else:
            if torch.any(valid_targets.sum(dim=-1) != 1):
                raise ValueError(
                    "Categorical morphology targets must activate exactly one label."
                )

            target_ids = feature_targets.to(torch.long).argmax(dim=-1)
            feature_per_token = functional.cross_entropy(
                feature_logits.transpose(1, 2),
                target_ids,
                weight=feature_weights,
                reduction="none",
            )

        morphology_feature_losses.append(
            _masked_mean(
                feature_per_token,
                targets.token_mask,
            )
        )

    morphology_loss = torch.stack(morphology_feature_losses).mean()

    lemma_per_token = functional.cross_entropy(
        logits.lemma_rule_logits.transpose(1, 2),
        targets.lemma_rule_ids,
        reduction="none",
    )
    lemma_rule_loss = _masked_mean(
        lemma_per_token,
        targets.token_mask & targets.lemma_rule_mask,
    )

    morphology_bundle_result: MorphologyBundleLossResult | None = None
    if morphology_bundle_loss_policy is not None:
        if logits.morphology_bundle_scores is None:
            raise ValueError("Morphology bundle loss requires bundle candidate scores.")
        bundle_loss_scores = (
            logits.morphology_bundle_scores
            if logits.morphology_bundle_loss_scores is None
            else logits.morphology_bundle_loss_scores
        )
        morphology_bundle_result = calculate_morphology_bundle_loss(
            candidate_scores=bundle_loss_scores,
            morphology_targets=targets.morphology_targets,
            token_mask=targets.token_mask,
            policy=morphology_bundle_loss_policy,
        )

    morphology_bundle_loss = (
        None if morphology_bundle_result is None else morphology_bundle_result.loss
    )
    morphology_bundle_loss_weight = (
        0.0
        if morphology_bundle_loss_policy is None
        else morphology_bundle_loss_policy.weight
    )
    total_loss = upos_loss + morphology_loss + lemma_rule_loss
    if morphology_bundle_loss is not None:
        total_loss = total_loss + morphology_bundle_loss_weight * morphology_bundle_loss

    return TokenTaskLosses(
        upos_loss=upos_loss,
        morphology_loss=morphology_loss,
        lemma_rule_loss=lemma_rule_loss,
        total_loss=total_loss,
        morphology_bundle_loss=morphology_bundle_loss,
        morphology_bundle_target_count=(
            0
            if morphology_bundle_result is None
            else morphology_bundle_result.target_count
        ),
        morphology_bundle_token_count=(
            0
            if morphology_bundle_result is None
            else morphology_bundle_result.token_count
        ),
        morphology_bundle_loss_weight=morphology_bundle_loss_weight,
    )
