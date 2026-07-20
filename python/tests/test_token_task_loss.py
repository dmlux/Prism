import math

import torch

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenTaskLogits
from prism.training import TokenTaskLossWeights, compute_token_task_loss


def test_token_task_loss_combines_all_training_tasks() -> None:
    logits = TokenTaskLogits(
        upos_logits=torch.zeros(
            (1, 2, 2),
            requires_grad=True,
        ),
        morphology_logits=(torch.zeros((1, 2, 2), requires_grad=True),),
        lemma_rule_logits=torch.zeros(
            (1, 2, 2),
            requires_grad=True,
        ),
    )
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor(
            [[1, 0]],
            dtype=torch.long,
        ),
        morphology_targets=(
            torch.tensor(
                [[[False, True], [False, False]]],
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor(
            [[1, 0]],
            dtype=torch.long,
        ),
        lemma_rule_mask=torch.tensor(
            [[True, False]],
            dtype=torch.bool,
        ),
        token_mask=torch.tensor(
            [[True, False]],
            dtype=torch.bool,
        ),
    )

    losses = compute_token_task_loss(
        logits=logits,
        targets=targets,
    )

    expected_task_loss = torch.tensor(math.log(2.0))

    torch.testing.assert_close(
        losses.upos_loss,
        expected_task_loss,
    )
    torch.testing.assert_close(
        losses.morphology_loss,
        expected_task_loss,
    )
    torch.testing.assert_close(
        losses.lemma_rule_loss,
        expected_task_loss,
    )
    torch.testing.assert_close(
        losses.total_loss,
        expected_task_loss * 3,
    )

    losses.total_loss.backward()

    assert logits.upos_logits.grad is not None
    assert logits.morphology_logits[0].grad is not None
    assert logits.lemma_rule_logits.grad is not None

    torch.testing.assert_close(
        logits.upos_logits.grad[0, 1],
        torch.zeros(2),
    )
    torch.testing.assert_close(
        logits.morphology_logits[0].grad[0, 1],
        torch.zeros(2),
    )
    torch.testing.assert_close(
        logits.lemma_rule_logits.grad[0, 1],
        torch.zeros(2),
    )


def test_token_task_loss_applies_morphology_positive_weights() -> None:
    logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2)),
        morphology_logits=(torch.zeros((1, 1, 2)),),
        lemma_rule_logits=torch.zeros((1, 1, 2)),
    )
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor([[0]]),
        morphology_targets=(
            torch.tensor(
                [[[False, True]]],
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor([[0]]),
        lemma_rule_mask=torch.tensor([[True]]),
        token_mask=torch.tensor([[True]]),
    )
    loss_weights = TokenTaskLossWeights(
        morphology_positive_weights=(torch.tensor([1.0, 3.0]),),
    )

    losses = compute_token_task_loss(
        logits=logits,
        targets=targets,
        loss_weights=loss_weights,
    )

    torch.testing.assert_close(
        losses.morphology_loss,
        torch.tensor(2.0 * math.log(2.0)),
    )
