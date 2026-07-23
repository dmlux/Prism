import torch
from torch import nn
from torch.optim import Optimizer

from prism.modeling import TokenizedBatch, TokenTaskLogits
from prism.schema import MorphologySchema
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.distillation import (
    CombinedTokenTaskLosses,
    TokenTaskDistillationPolicy,
    combine_token_task_losses,
    compute_token_task_distillation_loss,
)
from prism.training.losses import (
    MorphologyBundleLossPolicy,
    TokenTaskLosses,
    TokenTaskLossWeights,
    compute_token_task_loss,
)


def train_supervised_token_task_step(
    *,
    model: nn.Module,
    batch: SupervisedTokenTaskBatch,
    optimizer: Optimizer,
    max_gradient_norm: float,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
) -> TokenTaskLosses:
    if max_gradient_norm <= 0.0:
        raise ValueError("Maximum gradient norm must be positive.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = _forward_token_task_model(model=model, batch=batch)

    if not isinstance(logits, TokenTaskLogits):
        raise TypeError("Token-task model must return TokenTaskLogits.")

    losses = compute_token_task_loss(
        logits=logits,
        targets=batch.targets,
        morphology_schema=morphology_schema,
        loss_weights=loss_weights,
        morphology_bundle_loss_policy=morphology_bundle_loss_policy,
    )

    losses.total_loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_gradient_norm,
    )
    optimizer.step()

    return TokenTaskLosses(
        upos_loss=losses.upos_loss.detach(),
        morphology_loss=losses.morphology_loss.detach(),
        lemma_rule_loss=losses.lemma_rule_loss.detach(),
        total_loss=losses.total_loss.detach(),
        morphology_bundle_loss=(
            None
            if losses.morphology_bundle_loss is None
            else losses.morphology_bundle_loss.detach()
        ),
        morphology_bundle_target_count=losses.morphology_bundle_target_count,
        morphology_bundle_token_count=losses.morphology_bundle_token_count,
        morphology_bundle_loss_weight=losses.morphology_bundle_loss_weight,
    )


def evaluate_supervised_token_task_step(
    *,
    model: nn.Module,
    batch: SupervisedTokenTaskBatch,
    morphology_schema: MorphologySchema,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
) -> tuple[TokenTaskLogits, TokenTaskLosses]:
    model.eval()

    with torch.inference_mode():
        logits = _forward_token_task_model(model=model, batch=batch)

        if not isinstance(logits, TokenTaskLogits):
            raise TypeError("Token-task model must return TokenTaskLogits.")

        losses = compute_token_task_loss(
            logits=logits,
            targets=batch.targets,
            morphology_schema=morphology_schema,
            morphology_bundle_loss_policy=morphology_bundle_loss_policy,
        )

        return logits, losses


def train_distilled_token_task_step(
    *,
    student: nn.Module,
    teacher: nn.Module,
    batch: SupervisedTokenTaskBatch,
    optimizer: Optimizer,
    max_gradient_norm: float,
    distillation_policy: TokenTaskDistillationPolicy,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
    teacher_model_inputs: TokenizedBatch | None = None,
    morphology_bundle_loss_policy: MorphologyBundleLossPolicy | None = None,
) -> CombinedTokenTaskLosses:
    if max_gradient_norm <= 0.0:
        raise ValueError("Maximum gradient norm must be positive.")

    student.train()
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)

    resolved_teacher_inputs = (
        batch.model_inputs if teacher_model_inputs is None else teacher_model_inputs
    )

    with torch.no_grad():
        teacher_logits = _forward_token_task_model(
            model=teacher,
            batch=batch,
            model_inputs=resolved_teacher_inputs,
        )

    student_logits = _forward_token_task_model(model=student, batch=batch)

    if not isinstance(teacher_logits, TokenTaskLogits):
        raise TypeError("Teacher token-task model must return TokenTaskLogits.")
    if not isinstance(student_logits, TokenTaskLogits):
        raise TypeError("Student token-task model must return TokenTaskLogits.")

    supervised_losses = compute_token_task_loss(
        logits=student_logits,
        targets=batch.targets,
        morphology_schema=morphology_schema,
        loss_weights=loss_weights,
        morphology_bundle_loss_policy=morphology_bundle_loss_policy,
    )
    distillation_losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=batch.targets.token_mask,
        lemma_rule_mask=batch.targets.lemma_rule_mask,
        policy=distillation_policy,
        morphology_schema=morphology_schema,
        upos_ids=batch.targets.upos_ids,
        morphology_targets=batch.targets.morphology_targets,
        lemma_rule_ids=batch.targets.lemma_rule_ids,
        loss_weights=loss_weights,
    )
    losses = combine_token_task_losses(
        supervised_losses=supervised_losses,
        distillation_losses=distillation_losses,
        policy=distillation_policy,
    )

    losses.total_loss.backward()
    nn.utils.clip_grad_norm_(
        student.parameters(),
        max_norm=max_gradient_norm,
    )
    optimizer.step()

    def detached(
        task_losses: TokenTaskLosses,
    ) -> TokenTaskLosses:
        return TokenTaskLosses(
            upos_loss=task_losses.upos_loss.detach(),
            morphology_loss=(task_losses.morphology_loss.detach()),
            lemma_rule_loss=(task_losses.lemma_rule_loss.detach()),
            total_loss=task_losses.total_loss.detach(),
            morphology_bundle_loss=(
                None
                if task_losses.morphology_bundle_loss is None
                else task_losses.morphology_bundle_loss.detach()
            ),
            morphology_bundle_target_count=(task_losses.morphology_bundle_target_count),
            morphology_bundle_token_count=(task_losses.morphology_bundle_token_count),
            morphology_bundle_loss_weight=(task_losses.morphology_bundle_loss_weight),
        )

    return CombinedTokenTaskLosses(
        supervised_losses=detached(supervised_losses),
        distillation_losses=detached(distillation_losses),
        total_loss=losses.total_loss.detach(),
    )


def _forward_token_task_model(
    *,
    model: nn.Module,
    batch: SupervisedTokenTaskBatch,
    model_inputs: TokenizedBatch | None = None,
) -> object:
    resolved_model_inputs = batch.model_inputs if model_inputs is None else model_inputs
    if getattr(model, "character_encoder", None) is None:
        return model(resolved_model_inputs)

    if batch.character_inputs is None:
        raise ValueError("Character-aware token-task model requires character inputs.")

    return model(resolved_model_inputs, batch.character_inputs)
