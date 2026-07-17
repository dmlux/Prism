from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.steps import (
    train_supervised_token_task_step,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SupervisedEpochMetrics:
    batch_count: int
    token_count: int
    lemma_target_count: int
    upos_loss: float
    morphology_loss: float
    lemma_rule_loss: float

    @property
    def total_loss(self) -> float:
        return self.upos_loss + self.morphology_loss + self.lemma_rule_loss


def train_supervised_token_task_epoch(
    *,
    model: nn.Module,
    batches: Iterable[SupervisedTokenTaskBatch],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    max_gradient_norm: float,
) -> SupervisedEpochMetrics:
    model.to(device)

    batch_count = 0
    token_count = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )
    lemma_target_count = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )
    upos_loss_sum = torch.zeros((), device=device)
    morphology_loss_sum = torch.zeros((), device=device)
    lemma_rule_loss_sum = torch.zeros((), device=device)

    for batch in batches:
        device_batch = batch.to(device)
        losses = train_supervised_token_task_step(
            model=model,
            batch=device_batch,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
        )
        scheduler.step()

        current_token_count = device_batch.targets.token_mask.sum()
        current_lemma_target_count = (
            device_batch.targets.token_mask & device_batch.targets.lemma_rule_mask
        ).sum()

        batch_count += 1
        token_count += current_token_count
        lemma_target_count += current_lemma_target_count
        upos_loss_sum += losses.upos_loss * current_token_count
        morphology_loss_sum += losses.morphology_loss * current_token_count
        lemma_rule_loss_sum += losses.lemma_rule_loss * current_lemma_target_count

    if batch_count == 0:
        raise ValueError("Training epoch must contain batches.")

    token_count_value = int(token_count.item())
    lemma_target_count_value = int(lemma_target_count.item())

    if lemma_target_count_value == 0:
        lemma_rule_loss = 0.0
    else:
        lemma_rule_loss = (lemma_rule_loss_sum / lemma_target_count).item()

    return SupervisedEpochMetrics(
        batch_count=batch_count,
        token_count=token_count_value,
        lemma_target_count=lemma_target_count_value,
        upos_loss=(upos_loss_sum / token_count).item(),
        morphology_loss=(morphology_loss_sum / token_count).item(),
        lemma_rule_loss=lemma_rule_loss,
    )
