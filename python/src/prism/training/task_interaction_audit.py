"""Evaluation-only diagnostics for token-task ranking and gradient interaction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from prism.modeling import TokenTaskLogits
from prism.modeling.decoding import (
    MorphologyLogitCorrection,
    apply_morphology_logit_correction,
    decode_token_task_logits,
)
from prism.schema import TokenTaskSchema
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.losses import (
    TokenTaskLossWeights,
    compute_token_task_loss,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RankAuditMetrics:
    count: int
    top1_accuracy: float
    top2_accuracy: float
    top5_accuracy: float
    mean_rank: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedRankAuditMetrics:
    name: str
    metrics: RankAuditMetrics


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleRankingAudit:
    token_count: int
    candidate_covered_count: int
    final_bundle_correct_count: int
    final_bundle_error_count: int
    uncovered_error_count: int
    ranking_error_count: int
    refinement_error_count: int
    covered_ranks: RankAuditMetrics
    error_ranks: RankAuditMetrics | None
    mean_gold_margin: float
    mean_error_gold_margin: float | None

    @property
    def candidate_coverage(self) -> float:
        return self.candidate_covered_count / self.token_count


@dataclass(frozen=True, slots=True, kw_only=True)
class LemmaRuleRankingAudit:
    overall: RankAuditMetrics
    by_rule_frequency: tuple[NamedRankAuditMetrics, ...]
    by_token_frequency: tuple[NamedRankAuditMetrics, ...]
    by_upos: tuple[NamedRankAuditMetrics, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class GradientConflictPairAudit:
    first_task: str
    second_task: str
    sample_count: int
    conflict_count: int
    conflict_rate: float
    mean_cosine_similarity: float
    minimum_cosine_similarity: float
    maximum_cosine_similarity: float
    mean_first_gradient_norm: float
    mean_second_gradient_norm: float


@dataclass(frozen=True, slots=True, kw_only=True)
class GradientParameterGroupAudit:
    name: str
    parameter_count: int
    task_pairs: tuple[GradientConflictPairAudit, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskInteractionAudit:
    gradient_objective: str
    gradient_batch_indices: tuple[int, ...]
    morphology_bundles: MorphologyBundleRankingAudit
    lemma_rules: LemmaRuleRankingAudit
    gradient_groups: tuple[GradientParameterGroupAudit, ...]


class _RankAccumulator:
    def __init__(self) -> None:
        self.ranks: list[int] = []

    def add(self, ranks: Tensor) -> None:
        self.ranks.extend(int(rank) for rank in ranks.detach().cpu().tolist())

    def finish(self) -> RankAuditMetrics:
        if not self.ranks:
            raise ValueError("Rank audit requires at least one ranked target.")
        count = len(self.ranks)
        return RankAuditMetrics(
            count=count,
            top1_accuracy=sum(rank <= 1 for rank in self.ranks) / count,
            top2_accuracy=sum(rank <= 2 for rank in self.ranks) / count,
            top5_accuracy=sum(rank <= 5 for rank in self.ranks) / count,
            mean_rank=sum(self.ranks) / count,
            mean_reciprocal_rank=(
                sum(1.0 / rank for rank in self.ranks) / count
            ),
        )


class TokenTaskRankAuditAccumulator:
    """Accumulate complete-bundle and lemma-rule ranks without changing outputs."""

    def __init__(
        self,
        *,
        schema: TokenTaskSchema,
        candidate_morphology_targets: tuple[Tensor, ...],
        lemma_rule_training_counts: Sequence[int],
    ) -> None:
        if len(candidate_morphology_targets) != len(schema.morphology.features):
            raise ValueError("Bundle candidates must match the morphology schema.")
        if not candidate_morphology_targets:
            raise ValueError("Bundle rank audit requires candidates.")
        candidate_count = candidate_morphology_targets[0].shape[0]
        if candidate_count <= 0 or any(
            targets.ndim != 2 or targets.shape[0] != candidate_count
            for targets in candidate_morphology_targets
        ):
            raise ValueError("Bundle candidate targets must share candidates.")
        if any(targets.dtype != torch.bool for targets in candidate_morphology_targets):
            raise ValueError("Bundle candidate targets must use torch.bool.")
        if len(lemma_rule_training_counts) != len(schema.lemma_rules.rules):
            raise ValueError("Lemma-rule counts must match the lemma schema.")
        if any(count <= 0 for count in lemma_rule_training_counts):
            raise ValueError("Every schema lemma rule must occur in training.")

        self.schema = schema
        self.candidate_morphology_targets = tuple(
            targets.detach().cpu() for targets in candidate_morphology_targets
        )
        self.lemma_rule_training_counts = tuple(lemma_rule_training_counts)
        self.bundle_token_count = 0
        self.bundle_candidate_covered_count = 0
        self.bundle_final_correct_count = 0
        self.bundle_final_error_count = 0
        self.bundle_uncovered_error_count = 0
        self.bundle_ranking_error_count = 0
        self.bundle_refinement_error_count = 0
        self.bundle_covered_ranks = _RankAccumulator()
        self.bundle_error_ranks = _RankAccumulator()
        self.bundle_gold_margins: list[float] = []
        self.bundle_error_gold_margins: list[float] = []
        self.lemma_overall = _RankAccumulator()
        self.lemma_by_rule_frequency: dict[str, _RankAccumulator] = {}
        self.lemma_by_token_frequency: dict[str, _RankAccumulator] = {}
        self.lemma_by_upos: dict[str, _RankAccumulator] = {}

    def add(
        self,
        *,
        logits: TokenTaskLogits,
        targets: object,
        rare_mask: Tensor,
        oov_mask: Tensor,
    ) -> None:
        from prism.data import TokenTaskTargetBatch

        if not isinstance(targets, TokenTaskTargetBatch):
            raise TypeError("Rank audit targets must use TokenTaskTargetBatch.")
        if logits.morphology_bundle_scores is None:
            raise ValueError("Bundle rank audit requires candidate scores.")
        if rare_mask.shape != targets.token_mask.shape:
            raise ValueError("Rare mask must match token dimensions.")
        if oov_mask.shape != targets.token_mask.shape:
            raise ValueError("OOV mask must match token dimensions.")
        if rare_mask.dtype != torch.bool or oov_mask.dtype != torch.bool:
            raise ValueError("Frequency masks must use torch.bool.")
        if (rare_mask & oov_mask).any().item():
            raise ValueError("Rare and OOV masks must not overlap.")

        self._add_bundle_ranks(
            logits=logits,
            targets=targets,
        )
        self._add_lemma_ranks(
            logits=logits,
            targets=targets,
            rare_mask=rare_mask.to(device=targets.token_mask.device),
            oov_mask=oov_mask.to(device=targets.token_mask.device),
        )

    def _add_bundle_ranks(
        self,
        *,
        logits: TokenTaskLogits,
        targets: object,
    ) -> None:
        from prism.data import TokenTaskTargetBatch

        if not isinstance(targets, TokenTaskTargetBatch):
            raise TypeError("Bundle audit targets must use TokenTaskTargetBatch.")
        candidate_scores = logits.morphology_bundle_scores
        if candidate_scores is None:
            raise ValueError("Bundle rank audit requires candidate scores.")

        matching_candidates = torch.ones(
            candidate_scores.shape,
            dtype=torch.bool,
            device=candidate_scores.device,
        )
        for feature_targets, candidate_targets in zip(
            targets.morphology_targets,
            self.candidate_morphology_targets,
            strict=True,
        ):
            resolved_candidates = candidate_targets.to(device=candidate_scores.device)
            matching_candidates &= (
                feature_targets.unsqueeze(-2) == resolved_candidates.unsqueeze(0).unsqueeze(0)
            ).all(dim=-1)

        token_mask = targets.token_mask
        covered = token_mask & matching_candidates.any(dim=-1)
        best_gold_scores = candidate_scores.masked_fill(
            ~matching_candidates,
            -torch.inf,
        ).max(dim=-1).values
        ranks = (
            (candidate_scores > best_gold_scores.unsqueeze(-1)).sum(dim=-1) + 1
        )
        best_non_gold_scores = candidate_scores.masked_fill(
            matching_candidates,
            -torch.inf,
        ).max(dim=-1).values
        margins = best_gold_scores - best_non_gold_scores

        bundle_correct = token_mask.clone()
        for predictions, feature_targets in zip(
            decode_token_task_logits(
                logits=logits,
                token_mask=token_mask,
                morphology_schema=self.schema.morphology,
            ).morphology_predictions,
            targets.morphology_targets,
            strict=True,
        ):
            bundle_correct &= (predictions == feature_targets).all(dim=-1)

        final_errors = token_mask & ~bundle_correct
        covered_errors = final_errors & covered
        ranking_errors = covered_errors & (ranks > 1)
        refinement_errors = covered_errors & (ranks == 1)

        self.bundle_token_count += int(token_mask.sum().item())
        self.bundle_candidate_covered_count += int(covered.sum().item())
        self.bundle_final_correct_count += int(bundle_correct.sum().item())
        self.bundle_final_error_count += int(final_errors.sum().item())
        self.bundle_uncovered_error_count += int((final_errors & ~covered).sum().item())
        self.bundle_ranking_error_count += int(ranking_errors.sum().item())
        self.bundle_refinement_error_count += int(refinement_errors.sum().item())
        self.bundle_covered_ranks.add(ranks[covered])
        if covered_errors.any().item():
            self.bundle_error_ranks.add(ranks[covered_errors])
        self.bundle_gold_margins.extend(
            float(value) for value in margins[covered].detach().cpu().tolist()
        )
        self.bundle_error_gold_margins.extend(
            float(value)
            for value in margins[covered_errors].detach().cpu().tolist()
        )

    def _add_lemma_ranks(
        self,
        *,
        logits: TokenTaskLogits,
        targets: object,
        rare_mask: Tensor,
        oov_mask: Tensor,
    ) -> None:
        from prism.data import TokenTaskTargetBatch

        if not isinstance(targets, TokenTaskTargetBatch):
            raise TypeError("Lemma audit targets must use TokenTaskTargetBatch.")
        lemma_mask = targets.token_mask & targets.lemma_rule_mask
        gold_scores = logits.lemma_rule_logits.gather(
            dim=-1,
            index=targets.lemma_rule_ids.unsqueeze(-1),
        ).squeeze(-1)
        ranks = (
            (logits.lemma_rule_logits > gold_scores.unsqueeze(-1)).sum(dim=-1) + 1
        )
        selected_ranks = ranks[lemma_mask]
        self.lemma_overall.add(selected_ranks)

        selected_rule_ids = targets.lemma_rule_ids[lemma_mask].detach().cpu().tolist()
        selected_upos_ids = targets.upos_ids[lemma_mask].detach().cpu().tolist()
        selected_rare = rare_mask[lemma_mask].detach().cpu().tolist()
        selected_oov = oov_mask[lemma_mask].detach().cpu().tolist()
        selected_rank_values = selected_ranks.detach().cpu().tolist()

        for rank, rule_id, upos_id, is_rare, is_oov in zip(
            selected_rank_values,
            selected_rule_ids,
            selected_upos_ids,
            selected_rare,
            selected_oov,
            strict=True,
        ):
            rule_frequency = self.lemma_rule_training_counts[rule_id]
            rule_group = _lemma_rule_frequency_group(rule_frequency)
            token_group = "oov" if is_oov else "rare" if is_rare else "frequent"
            upos_label = self.schema.upos.labels[upos_id]
            _add_scalar_rank(self.lemma_by_rule_frequency, rule_group, rank)
            _add_scalar_rank(self.lemma_by_token_frequency, token_group, rank)
            _add_scalar_rank(self.lemma_by_upos, upos_label, rank)

    def finish(self) -> tuple[MorphologyBundleRankingAudit, LemmaRuleRankingAudit]:
        if self.bundle_token_count == 0:
            raise ValueError("Task rank audit requires tokens.")
        if not self.bundle_gold_margins:
            raise ValueError("Task rank audit requires covered bundle candidates.")

        bundle = MorphologyBundleRankingAudit(
            token_count=self.bundle_token_count,
            candidate_covered_count=self.bundle_candidate_covered_count,
            final_bundle_correct_count=self.bundle_final_correct_count,
            final_bundle_error_count=self.bundle_final_error_count,
            uncovered_error_count=self.bundle_uncovered_error_count,
            ranking_error_count=self.bundle_ranking_error_count,
            refinement_error_count=self.bundle_refinement_error_count,
            covered_ranks=self.bundle_covered_ranks.finish(),
            error_ranks=(
                None
                if not self.bundle_error_ranks.ranks
                else self.bundle_error_ranks.finish()
            ),
            mean_gold_margin=(
                sum(self.bundle_gold_margins) / len(self.bundle_gold_margins)
            ),
            mean_error_gold_margin=(
                None
                if not self.bundle_error_gold_margins
                else (
                    sum(self.bundle_error_gold_margins)
                    / len(self.bundle_error_gold_margins)
                )
            ),
        )
        lemma = LemmaRuleRankingAudit(
            overall=self.lemma_overall.finish(),
            by_rule_frequency=_finish_named_ranks(
                self.lemma_by_rule_frequency,
                preferred_order=("singleton", "2-5", "6-20", "21+"),
            ),
            by_token_frequency=_finish_named_ranks(
                self.lemma_by_token_frequency,
                preferred_order=("frequent", "rare", "oov"),
            ),
            by_upos=_finish_named_ranks(
                self.lemma_by_upos,
                preferred_order=self.schema.upos.labels,
            ),
        )
        return bundle, lemma


class GradientConflictAuditAccumulator:
    """Measure pairwise task-gradient alignment on shared parameter groups."""

    _TASK_PAIRS = (
        ("upos", "morphology"),
        ("upos", "lemma"),
        ("morphology", "lemma"),
    )

    def __init__(
        self,
        *,
        parameter_groups: Mapping[str, Sequence[nn.Parameter]],
    ) -> None:
        resolved_groups = {
            name: tuple(parameter for parameter in parameters if parameter.requires_grad)
            for name, parameters in parameter_groups.items()
        }
        if not resolved_groups or any(not parameters for parameters in resolved_groups.values()):
            raise ValueError("Gradient audit groups must contain trainable parameters.")

        parameter_indices: dict[int, int] = {}
        all_parameters: list[nn.Parameter] = []
        self.group_indices: dict[str, tuple[int, ...]] = {}
        self.group_parameter_counts: dict[str, int] = {}
        for name, parameters in resolved_groups.items():
            indices: list[int] = []
            for parameter in parameters:
                identity = id(parameter)
                if identity not in parameter_indices:
                    parameter_indices[identity] = len(all_parameters)
                    all_parameters.append(parameter)
                indices.append(parameter_indices[identity])
            self.group_indices[name] = tuple(indices)
            self.group_parameter_counts[name] = sum(
                parameter.numel() for parameter in parameters
            )
        self.parameters = tuple(all_parameters)
        self.samples: dict[
            tuple[str, str, str],
            list[tuple[float, float, float]],
        ] = {}

    def add(
        self,
        *,
        upos_loss: Tensor,
        morphology_loss: Tensor,
        lemma_loss: Tensor,
    ) -> None:
        task_losses = {
            "upos": upos_loss,
            "morphology": morphology_loss,
            "lemma": lemma_loss,
        }
        task_gradients: dict[str, tuple[Tensor | None, ...]] = {}
        for task_index, (task_name, loss) in enumerate(task_losses.items()):
            task_gradients[task_name] = torch.autograd.grad(
                loss,
                self.parameters,
                retain_graph=task_index < len(task_losses) - 1,
                allow_unused=True,
            )

        for group_name, indices in self.group_indices.items():
            for first_task, second_task in self._TASK_PAIRS:
                first_gradients = task_gradients[first_task]
                second_gradients = task_gradients[second_task]
                dot = torch.zeros((), device=upos_loss.device)
                first_squared_norm = torch.zeros((), device=upos_loss.device)
                second_squared_norm = torch.zeros((), device=upos_loss.device)
                for index in indices:
                    first = first_gradients[index]
                    second = second_gradients[index]
                    if first is not None:
                        first_squared_norm += first.detach().square().sum()
                    if second is not None:
                        second_squared_norm += second.detach().square().sum()
                    if first is not None and second is not None:
                        dot += (first.detach() * second.detach()).sum()

                first_norm = first_squared_norm.sqrt()
                second_norm = second_squared_norm.sqrt()
                if first_norm.item() == 0.0 or second_norm.item() == 0.0:
                    continue
                cosine = dot / (first_norm * second_norm)
                self.samples.setdefault(
                    (group_name, first_task, second_task),
                    [],
                ).append(
                    (
                        float(cosine.item()),
                        float(first_norm.item()),
                        float(second_norm.item()),
                    )
                )

    def finish(self) -> tuple[GradientParameterGroupAudit, ...]:
        groups: list[GradientParameterGroupAudit] = []
        for group_name in self.group_indices:
            pairs: list[GradientConflictPairAudit] = []
            for first_task, second_task in self._TASK_PAIRS:
                samples = self.samples.get(
                    (group_name, first_task, second_task),
                    [],
                )
                if not samples:
                    continue
                cosines = [sample[0] for sample in samples]
                first_norms = [sample[1] for sample in samples]
                second_norms = [sample[2] for sample in samples]
                conflict_count = sum(cosine < 0.0 for cosine in cosines)
                pairs.append(
                    GradientConflictPairAudit(
                        first_task=first_task,
                        second_task=second_task,
                        sample_count=len(samples),
                        conflict_count=conflict_count,
                        conflict_rate=conflict_count / len(samples),
                        mean_cosine_similarity=sum(cosines) / len(cosines),
                        minimum_cosine_similarity=min(cosines),
                        maximum_cosine_similarity=max(cosines),
                        mean_first_gradient_norm=(
                            sum(first_norms) / len(first_norms)
                        ),
                        mean_second_gradient_norm=(
                            sum(second_norms) / len(second_norms)
                        ),
                    )
                )
            groups.append(
                GradientParameterGroupAudit(
                    name=group_name,
                    parameter_count=self.group_parameter_counts[group_name],
                    task_pairs=tuple(pairs),
                )
            )
        return tuple(groups)


def evenly_spaced_batch_indices(
    *,
    batch_count: int,
    selected_count: int,
) -> tuple[int, ...]:
    if batch_count <= 0:
        raise ValueError("Batch count must be positive.")
    if selected_count <= 0:
        raise ValueError("Selected batch count must be positive.")
    if selected_count >= batch_count:
        return tuple(range(batch_count))
    if selected_count == 1:
        return (batch_count // 2,)
    return tuple(
        round(index * (batch_count - 1) / (selected_count - 1))
        for index in range(selected_count)
    )


def token_task_shared_parameter_groups(
    model: nn.Module,
) -> dict[str, tuple[nn.Parameter, ...]]:
    backbone = getattr(model, "backbone", None)
    heads = getattr(model, "heads", None)
    if not isinstance(backbone, nn.Module) or not isinstance(heads, nn.Module):
        raise TypeError("Gradient audit requires a token tagger with backbone and heads.")

    groups: dict[str, tuple[nn.Parameter, ...]] = {}
    backbone_parameters = list(backbone.parameters())
    layer_aggregation = getattr(model, "layer_aggregation", None)
    if isinstance(layer_aggregation, nn.Module):
        backbone_parameters.extend(layer_aggregation.parameters())
    groups["backbone"] = _unique_parameters(backbone_parameters)

    input_projection = getattr(heads, "input_projection", None)
    if isinstance(input_projection, nn.Module):
        projection_parameters = _unique_parameters(input_projection.parameters())
        if projection_parameters:
            groups["shared_projection"] = projection_parameters

    character_modules = (
        getattr(model, "character_encoder", None),
        getattr(heads, "character_fusion", None),
    )
    character_parameters = _unique_parameters(
        parameter
        for module in character_modules
        if isinstance(module, nn.Module)
        for parameter in module.parameters()
    )
    if character_parameters:
        groups["character_path"] = character_parameters

    return groups


def audit_token_task_interactions(
    *,
    model: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    device: torch.device,
    schema: TokenTaskSchema,
    candidate_morphology_targets: tuple[Tensor, ...],
    lemma_rule_training_counts: Sequence[int],
    token_slice_masks: Mapping[str, Sequence[Tensor]],
    gradient_batch_indices: Sequence[int],
    loss_weights: TokenTaskLossWeights | None = None,
    morphology_logit_correction: MorphologyLogitCorrection | None = None,
) -> TokenTaskInteractionAudit:
    if "rare" not in token_slice_masks or "oov" not in token_slice_masks:
        raise ValueError("Task interaction audit requires Rare and OOV masks.")
    selected_gradient_batches = frozenset(gradient_batch_indices)
    if not selected_gradient_batches:
        raise ValueError("Gradient audit requires selected batches.")

    rank_accumulator = TokenTaskRankAuditAccumulator(
        schema=schema,
        candidate_morphology_targets=candidate_morphology_targets,
        lemma_rule_training_counts=lemma_rule_training_counts,
    )
    gradient_accumulator = GradientConflictAuditAccumulator(
        parameter_groups=token_task_shared_parameter_groups(model),
    )
    resolved_loss_weights = (
        None if loss_weights is None else loss_weights.to(device=device)
    )

    model.to(device)
    model.eval()
    batch_count = 0
    for batch_index, raw_batch in enumerate(batches):
        batch = raw_batch.to(device)
        if batch_index in selected_gradient_batches:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = _forward_token_task_model(model=model, batch=batch)
                losses = compute_token_task_loss(
                    logits=logits,
                    targets=batch.targets,
                    morphology_schema=schema.morphology,
                    loss_weights=resolved_loss_weights,
                )
                prediction_logits = _prediction_logits(
                    logits=logits,
                    schema=schema,
                    morphology_logit_correction=morphology_logit_correction,
                )
                rank_accumulator.add(
                    logits=prediction_logits,
                    targets=batch.targets,
                    rare_mask=token_slice_masks["rare"][batch_index],
                    oov_mask=token_slice_masks["oov"][batch_index],
                )
                gradient_accumulator.add(
                    upos_loss=losses.upos_loss,
                    morphology_loss=losses.morphology_loss,
                    lemma_loss=losses.lemma_rule_loss,
                )
        else:
            with torch.inference_mode():
                logits = _forward_token_task_model(model=model, batch=batch)
                rank_accumulator.add(
                    logits=_prediction_logits(
                        logits=logits,
                        schema=schema,
                        morphology_logit_correction=morphology_logit_correction,
                    ),
                    targets=batch.targets,
                    rare_mask=token_slice_masks["rare"][batch_index],
                    oov_mask=token_slice_masks["oov"][batch_index],
                )
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Task interaction audit requires batches.")
    for name in ("rare", "oov"):
        if len(token_slice_masks[name]) != batch_count:
            raise ValueError(f"{name.title()} masks must match audit batches.")
    if max(selected_gradient_batches) >= batch_count:
        raise ValueError("Gradient batch index is out of range.")

    morphology_bundles, lemma_rules = rank_accumulator.finish()
    model.zero_grad(set_to_none=True)
    return TokenTaskInteractionAudit(
        gradient_objective="checkpoint-weighted-supervised",
        gradient_batch_indices=tuple(sorted(selected_gradient_batches)),
        morphology_bundles=morphology_bundles,
        lemma_rules=lemma_rules,
        gradient_groups=gradient_accumulator.finish(),
    )


def _prediction_logits(
    *,
    logits: TokenTaskLogits,
    schema: TokenTaskSchema,
    morphology_logit_correction: MorphologyLogitCorrection | None,
) -> TokenTaskLogits:
    if morphology_logit_correction is None:
        return logits
    return apply_morphology_logit_correction(
        logits=logits,
        morphology_schema=schema.morphology,
        correction=morphology_logit_correction,
    )


def _forward_token_task_model(
    *,
    model: nn.Module,
    batch: SupervisedTokenTaskBatch,
) -> TokenTaskLogits:
    if getattr(model, "character_encoder", None) is None:
        logits = model(batch.model_inputs)
    else:
        if batch.character_inputs is None:
            raise ValueError("Character-aware audit requires character inputs.")
        logits = model(batch.model_inputs, batch.character_inputs)
    if not isinstance(logits, TokenTaskLogits):
        raise TypeError("Task interaction audit requires TokenTaskLogits.")
    return logits


def _lemma_rule_frequency_group(count: int) -> str:
    if count == 1:
        return "singleton"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return "21+"


def _add_scalar_rank(
    groups: dict[str, _RankAccumulator],
    name: str,
    rank: int,
) -> None:
    accumulator = groups.setdefault(name, _RankAccumulator())
    accumulator.ranks.append(rank)


def _finish_named_ranks(
    groups: Mapping[str, _RankAccumulator],
    *,
    preferred_order: Sequence[str],
) -> tuple[NamedRankAuditMetrics, ...]:
    order = {name: index for index, name in enumerate(preferred_order)}
    return tuple(
        NamedRankAuditMetrics(
            name=name,
            metrics=accumulator.finish(),
        )
        for name, accumulator in sorted(
            groups.items(),
            key=lambda item: (order.get(item[0], math.inf), item[0]),
        )
    )


def _unique_parameters(
    parameters: Iterable[nn.Parameter],
) -> tuple[nn.Parameter, ...]:
    seen: set[int] = set()
    resolved: list[nn.Parameter] = []
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        resolved.append(parameter)
    return tuple(resolved)
