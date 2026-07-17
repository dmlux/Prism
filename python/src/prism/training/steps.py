from torch import nn
from torch.optim import Optimizer

from prism.modeling import TokenTaskLogits
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.losses import (
    TokenTaskLosses,
    compute_token_task_loss,
)


def train_supervised_token_task_step(
    *,
    model: nn.Module,
    batch: SupervisedTokenTaskBatch,
    optimizer: Optimizer,
    max_gradient_norm: float,
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
