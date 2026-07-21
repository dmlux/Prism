import torch
from torch import nn
from torch.optim import Optimizer

from prism.modeling import TokenizedBatch, TokenTaskLogits
from prism.schema import MorphologySchema
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.distillation import (
    CombinedTokenTaskLosses,
    combine_token_task_losses,
    compute_token_task_distillation_loss,
)
from prism.training.losses import (
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
) -> TokenTaskLosses:
    if max_gradient_norm <= 0.0:
        raise ValueError("Maximum gradient norm must be positive.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(batch.model_inputs)

    if not isinstance(logits, TokenTaskLogits):
        raise TypeError("Token-task model must return TokenTaskLogits.")

    losses = compute_token_task_loss(
        logits=logits,
        targets=batch.targets,
        morphology_schema=morphology_schema,
        loss_weights=loss_weights,
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
    )


def evaluate_supervised_token_task_step(
    *,
    model: nn.Module,
    batch: SupervisedTokenTaskBatch,
    morphology_schema: MorphologySchema,
) -> tuple[TokenTaskLogits, TokenTaskLosses]:
    model.eval()

    with torch.inference_mode():
        logits = model(batch.model_inputs)

        if not isinstance(logits, TokenTaskLogits):
            raise TypeError("Token-task model must return TokenTaskLogits.")

        losses = compute_token_task_loss(
            logits=logits,
            targets=batch.targets,
            morphology_schema=morphology_schema,
        )

        return logits, losses


def train_distilled_token_task_step(
    *,
    student: nn.Module,
    teacher: nn.Module,
    batch: SupervisedTokenTaskBatch,
    optimizer: Optimizer,
    max_gradient_norm: float,
    temperature: float,
    distillation_weight: float,
    morphology_schema: MorphologySchema,
    loss_weights: TokenTaskLossWeights | None = None,
    teacher_model_inputs: TokenizedBatch | None = None,
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
        teacher_logits = teacher(resolved_teacher_inputs)

    student_logits = student(batch.model_inputs)

    if not isinstance(teacher_logits, TokenTaskLogits):
        raise TypeError("Teacher token-task model must return TokenTaskLogits.")
    if not isinstance(student_logits, TokenTaskLogits):
        raise TypeError("Student token-task model must return TokenTaskLogits.")

    supervised_losses = compute_token_task_loss(
        logits=student_logits,
        targets=batch.targets,
        morphology_schema=morphology_schema,
        loss_weights=loss_weights,
    )
    distillation_losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=batch.targets.token_mask,
        lemma_rule_mask=batch.targets.lemma_rule_mask,
        temperature=temperature,
        morphology_schema=morphology_schema,
        morphology_targets=batch.targets.morphology_targets,
        loss_weights=loss_weights,
    )
    losses = combine_token_task_losses(
        supervised_losses=supervised_losses,
        distillation_losses=distillation_losses,
        distillation_weight=distillation_weight,
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
        )

    return CombinedTokenTaskLosses(
        supervised_losses=detached(supervised_losses),
        distillation_losses=detached(distillation_losses),
        total_loss=losses.total_loss.detach(),
    )
