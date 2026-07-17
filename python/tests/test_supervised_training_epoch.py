import math

import torch
from torch import nn

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenizedBatch, TokenTaskLogits
from prism.training import (
    SupervisedTokenTaskBatch,
    build_linear_warmup_decay_scheduler,
    train_supervised_token_task_epoch,
)


class TinyEpochModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.upos = nn.Parameter(torch.zeros(2))
        self.morphology = nn.Parameter(torch.zeros(2))
        self.lemma_rules = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        batch: TokenizedBatch,
    ) -> TokenTaskLogits:
        dimensions = (
            batch.batch_size,
            batch.max_token_count,
            2,
        )

        return TokenTaskLogits(
            upos_logits=self.upos.expand(dimensions),
            morphology_logits=(self.morphology.expand(dimensions),),
            lemma_rule_logits=self.lemma_rules.expand(dimensions),
        )


def _training_batch(target_id: int) -> SupervisedTokenTaskBatch:
    return SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor(
                [[1, 2, 3]],
                dtype=torch.long,
            ),
            attention_mask=torch.tensor(
                [[True, True, True]],
                dtype=torch.bool,
            ),
            first_subword_indices=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor(
                [[target_id]],
                dtype=torch.long,
            ),
            morphology_targets=(
                torch.tensor(
                    [[[target_id == 0, target_id == 1]]],
                    dtype=torch.bool,
                ),
            ),
            lemma_rule_ids=torch.tensor(
                [[target_id]],
                dtype=torch.long,
            ),
            lemma_rule_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
    )


def test_training_epoch_runs_all_batches_and_scheduler_steps() -> None:
    model = TinyEpochModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_step_count=2,
        warmup_ratio=0.0,
    )

    metrics = train_supervised_token_task_epoch(
        model=model,
        batches=(
            _training_batch(0),
            _training_batch(1),
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
    )

    assert metrics.batch_count == 2
    assert metrics.token_count == 2
    assert metrics.lemma_target_count == 2
    assert math.isfinite(metrics.upos_loss)
    assert math.isfinite(metrics.morphology_loss)
    assert math.isfinite(metrics.lemma_rule_loss)
    assert math.isfinite(metrics.total_loss)
    assert optimizer.param_groups[0]["lr"] == 0.0
