"""Differentiable complete-bundle refinement for morphology logits."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from prism.schema import MorphologySchema


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleCandidate:
    upos_id: int
    morphology: tuple[tuple[bool, ...], ...]
    training_count: int

    def __post_init__(self) -> None:
        if self.upos_id < 0:
            raise ValueError("Bundle-candidate UPOS ID must not be negative.")
        if not self.morphology:
            raise ValueError("Bundle candidate must contain morphology features.")
        if any(not labels or not any(labels) for labels in self.morphology):
            raise ValueError("Every bundle feature must activate at least one label.")
        if self.training_count <= 0:
            raise ValueError("Bundle-candidate training count must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleRerankerSpec:
    maximum_candidates_per_upos: int
    candidates: tuple[MorphologyBundleCandidate, ...]

    def __post_init__(self) -> None:
        if self.maximum_candidates_per_upos <= 0:
            raise ValueError("Maximum candidates per UPOS must be positive.")
        if not self.candidates:
            raise ValueError("Bundle reranker requires candidates.")

        candidate_counts: dict[int, int] = {}
        identities: set[tuple[int, tuple[tuple[bool, ...], ...]]] = set()
        for candidate in self.candidates:
            candidate_counts[candidate.upos_id] = (
                candidate_counts.get(candidate.upos_id, 0) + 1
            )
            identity = (candidate.upos_id, candidate.morphology)
            if identity in identities:
                raise ValueError("Bundle-reranker candidates must be unique.")
            identities.add(identity)

        if max(candidate_counts.values()) > self.maximum_candidates_per_upos:
            raise ValueError("Bundle-reranker candidate limit was exceeded.")


def validate_morphology_bundle_reranker_spec(
    *,
    spec: MorphologyBundleRerankerSpec,
    upos_label_count: int,
    morphology_schema: MorphologySchema,
) -> None:
    if upos_label_count <= 1:
        raise ValueError("UPOS label count must be greater than one.")

    for candidate in spec.candidates:
        if candidate.upos_id >= upos_label_count:
            raise ValueError("Bundle-candidate UPOS ID is out of range.")
        if len(candidate.morphology) != len(morphology_schema.features):
            raise ValueError("Bundle candidate must match the morphology schema.")
        for labels, feature in zip(
            candidate.morphology,
            morphology_schema.features,
            strict=True,
        ):
            if len(labels) != len(feature.labels):
                raise ValueError("Bundle candidate label count is invalid.")
            active_count = sum(labels)
            if labels[0] and active_count != 1:
                raise ValueError("The morphology NONE label must be exclusive.")
            if not feature.allows_multiple_values and active_count != 1:
                raise ValueError("Exclusive morphology features need one label.")


class MorphologyBundleReranker(nn.Module):
    """Blend complete-bundle evidence into independently predicted features.

    Candidate scores use soft UPOS evidence, all independent feature logits,
    and a small learned token-to-candidate residual. The output remains a
    residual refinement of the independent logits, so unseen combinations and
    UPOS uncertainty are never hard-blocked by the inventory.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        upos_label_count: int,
        morphology_schema: MorphologySchema,
        spec: MorphologyBundleRerankerSpec,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Hidden size must be positive.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Dropout probability must be in [0,1).")
        validate_morphology_bundle_reranker_spec(
            spec=spec,
            upos_label_count=upos_label_count,
            morphology_schema=morphology_schema,
        )

        self.upos_label_count = upos_label_count
        self.morphology_schema = morphology_schema
        self.spec = spec
        self.enabled = True
        self.isolate_direct_loss_gradient = False
        candidate_count = len(spec.candidates)

        self.register_buffer(
            "candidate_upos_ids",
            torch.tensor(
                [candidate.upos_id for candidate in spec.candidates],
                dtype=torch.long,
            ),
        )
        self._feature_count = len(morphology_schema.features)
        for index in range(self._feature_count):
            self.register_buffer(
                f"candidate_label_mask_{index}",
                torch.tensor(
                    [candidate.morphology[index] for candidate in spec.candidates],
                    dtype=torch.bool,
                ),
            )
        self.dropout = nn.Dropout(dropout_probability)
        self.candidate_projection = nn.Linear(hidden_size, candidate_count)
        self.refinement_gates = nn.Parameter(
            torch.full((len(morphology_schema.features),), -2.0)
        )
        nn.init.zeros_(self.candidate_projection.weight)
        nn.init.zeros_(self.candidate_projection.bias)

    @property
    def candidate_label_masks(self) -> tuple[Tensor, ...]:
        return tuple(
            getattr(self, f"candidate_label_mask_{index}")
            for index in range(self._feature_count)
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_direct_loss_gradient_isolation(self, enabled: bool) -> None:
        self.isolate_direct_loss_gradient = enabled

    def _candidate_evidence_scores(
        self,
        *,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> Tensor:
        scores = torch.log_softmax(upos_logits, dim=-1).index_select(
            -1,
            self.candidate_upos_ids,
        )

        for feature_logits, feature, candidate_labels in zip(
            morphology_logits,
            self.morphology_schema.features,
            self.candidate_label_masks,
            strict=True,
        ):
            if feature.allows_multiple_values:
                value_labels = candidate_labels[:, 1:]
                positive_scores = F.logsigmoid(feature_logits).unsqueeze(-2)
                negative_scores = F.logsigmoid(-feature_logits).unsqueeze(-2)
                scores = scores + torch.where(
                    value_labels,
                    positive_scores,
                    negative_scores,
                ).sum(dim=-1)
            else:
                label_ids = candidate_labels.to(dtype=torch.long).argmax(dim=-1)
                scores = scores + torch.log_softmax(
                    feature_logits,
                    dim=-1,
                ).index_select(-1, label_ids)

        return scores

    def _candidate_score_outputs(
        self,
        *,
        hidden_states: Tensor,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor | None]:
        evidence_scores = self._candidate_evidence_scores(
            upos_logits=upos_logits,
            morphology_logits=morphology_logits,
        )
        dropped_hidden_states = self.dropout(hidden_states)
        residual_scores = self.candidate_projection(dropped_hidden_states)
        candidate_scores = evidence_scores + residual_scores

        if not self.isolate_direct_loss_gradient or not torch.is_grad_enabled():
            return candidate_scores, None

        isolated_residual_scores = F.linear(
            dropped_hidden_states.detach(),
            self.candidate_projection.weight,
            self.candidate_projection.bias,
        )
        isolated_loss_scores = evidence_scores.detach() + isolated_residual_scores
        return candidate_scores, isolated_loss_scores

    def forward(
        self,
        *,
        hidden_states: Tensor,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        refined_logits, _ = self.refine_with_scores(
            hidden_states=hidden_states,
            upos_logits=upos_logits,
            morphology_logits=morphology_logits,
        )
        return refined_logits

    def refine_with_scores(
        self,
        *,
        hidden_states: Tensor,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[tuple[Tensor, ...], Tensor | None]:
        refined_logits, candidate_scores, _ = self.refine_with_training_scores(
            hidden_states=hidden_states,
            upos_logits=upos_logits,
            morphology_logits=morphology_logits,
        )
        return refined_logits, candidate_scores

    def refine_with_training_scores(
        self,
        *,
        hidden_states: Tensor,
        upos_logits: Tensor,
        morphology_logits: tuple[Tensor, ...],
    ) -> tuple[tuple[Tensor, ...], Tensor | None, Tensor | None]:
        if not self.enabled:
            return morphology_logits, None, None
        if upos_logits.ndim != 3 or upos_logits.shape[-1] != self.upos_label_count:
            raise ValueError("UPOS logits must match the reranker label count.")
        if hidden_states.shape[:2] != upos_logits.shape[:2]:
            raise ValueError("Reranker hidden states must match token dimensions.")
        if len(morphology_logits) != len(self.morphology_schema.features):
            raise ValueError("Morphology logits must match the reranker schema.")

        candidate_scores, isolated_loss_scores = self._candidate_score_outputs(
            hidden_states=hidden_states,
            upos_logits=upos_logits,
            morphology_logits=morphology_logits,
        )
        candidate_probabilities = torch.softmax(candidate_scores, dim=-1)
        refined_logits: list[Tensor] = []
        epsilon = torch.finfo(candidate_probabilities.dtype).eps

        for index, (feature_logits, feature, candidate_labels) in enumerate(
            zip(
                morphology_logits,
                self.morphology_schema.features,
                self.candidate_label_masks,
                strict=True,
            )
        ):
            label_probabilities = candidate_probabilities @ candidate_labels.to(
                dtype=candidate_probabilities.dtype
            )
            if feature.allows_multiple_values:
                value_probabilities = label_probabilities[..., 1:].clamp(
                    min=epsilon,
                    max=1.0 - epsilon,
                )
                bundle_logits = torch.logit(value_probabilities)
            else:
                bundle_logits = label_probabilities.clamp_min(epsilon).log()

            gate = torch.sigmoid(self.refinement_gates[index])
            refined_logits.append(feature_logits + gate * bundle_logits)

        return tuple(refined_logits), candidate_scores, isolated_loss_scores
