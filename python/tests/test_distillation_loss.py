import torch

from prism.modeling import TokenTaskLogits
from prism.training import (
    TokenTaskLossWeights,
    TokenTaskLosses,
    calculate_binary_distillation_loss,
    calculate_categorical_distillation_loss,
    combine_token_task_losses,
    compute_token_task_distillation_loss,
)


def test_categorical_distillation_ignores_masked_tokens() -> None:
    teacher_logits = torch.tensor(
        [
            [
                [4.0, 2.0, 0.0],
                [10.0, -10.0, 0.0],
            ]
        ]
    )
    student_logits = torch.tensor(
        [
            [
                [4.0, 2.0, 0.0],
                [-10.0, 10.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    token_mask = torch.tensor(
        [[True, False]],
        dtype=torch.bool,
    )

    loss = calculate_categorical_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        temperature=2.0,
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_binary_distillation_ignores_masked_tokens() -> None:
    teacher_logits = torch.tensor(
        [
            [
                [2.0, -1.0],
                [10.0, -10.0],
            ]
        ]
    )
    student_logits = torch.tensor(
        [
            [
                [2.0, -1.0],
                [-10.0, 10.0],
            ]
        ],
        requires_grad=True,
    )
    token_mask = torch.tensor(
        [[True, False]],
        dtype=torch.bool,
    )

    loss = calculate_binary_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        temperature=2.0,
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_token_task_distillation_combines_all_tasks() -> None:
    teacher_logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2)),
        morphology_logits=(torch.zeros((1, 1, 2)),),
        lemma_rule_logits=torch.zeros((1, 1, 2)),
    )
    student_logits = TokenTaskLogits(
        upos_logits=torch.tensor(
            [[[2.0, 0.0]]],
            requires_grad=True,
        ),
        morphology_logits=(
            torch.tensor(
                [[[1.0, -1.0]]],
                requires_grad=True,
            ),
        ),
        lemma_rule_logits=torch.tensor(
            [[[0.0, 2.0]]],
            requires_grad=True,
        ),
    )
    token_mask = torch.tensor(
        [[True]],
        dtype=torch.bool,
    )
    lemma_rule_mask = torch.tensor(
        [[True]],
        dtype=torch.bool,
    )

    losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=lemma_rule_mask,
        temperature=2.0,
    )

    torch.testing.assert_close(
        losses.total_loss,
        (losses.upos_loss + losses.morphology_loss + losses.lemma_rule_loss),
    )
    assert losses.total_loss.item() > 0.0


def test_combined_loss_preserves_gold_signal() -> None:
    supervised_losses = TokenTaskLosses(
        upos_loss=torch.tensor(1.0),
        morphology_loss=torch.tensor(2.0),
        lemma_rule_loss=torch.tensor(3.0),
        total_loss=torch.tensor(6.0),
    )
    distillation_losses = TokenTaskLosses(
        upos_loss=torch.tensor(0.25),
        morphology_loss=torch.tensor(0.5),
        lemma_rule_loss=torch.tensor(0.75),
        total_loss=torch.tensor(1.5),
    )

    losses = combine_token_task_losses(
        supervised_losses=supervised_losses,
        distillation_losses=distillation_losses,
        distillation_weight=0.5,
    )

    torch.testing.assert_close(
        losses.total_loss,
        torch.tensor(6.75),
    )
    assert losses.supervised_losses is supervised_losses
    assert losses.distillation_losses is distillation_losses


def test_distillation_only_backpropagates_into_student() -> None:
    teacher_logits = TokenTaskLogits(
        upos_logits=torch.zeros(
            (1, 1, 2),
            requires_grad=True,
        ),
        morphology_logits=(
            torch.zeros(
                (1, 1, 2),
                requires_grad=True,
            ),
        ),
        lemma_rule_logits=torch.zeros(
            (1, 1, 2),
            requires_grad=True,
        ),
    )
    student_logits = TokenTaskLogits(
        upos_logits=torch.tensor(
            [[[2.0, 0.0]]],
            requires_grad=True,
        ),
        morphology_logits=(
            torch.tensor(
                [[[1.0, -1.0]]],
                requires_grad=True,
            ),
        ),
        lemma_rule_logits=torch.tensor(
            [[[0.0, 2.0]]],
            requires_grad=True,
        ),
    )
    token_mask = torch.tensor(
        [[True]],
        dtype=torch.bool,
    )

    losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        temperature=2.0,
    )
    losses.total_loss.backward()

    assert student_logits.upos_logits.grad is not None
    assert student_logits.morphology_logits[0].grad is not None
    assert student_logits.lemma_rule_logits.grad is not None

    assert teacher_logits.upos_logits.grad is None
    assert teacher_logits.morphology_logits[0].grad is None
    assert teacher_logits.lemma_rule_logits.grad is None


def test_binary_distillation_weights_positive_targets() -> None:
    teacher_logits = torch.tensor(
        [[[2.0, 2.0]]],
    )
    student_logits = torch.tensor(
        [[[-2.0, -2.0]]],
        requires_grad=True,
    )
    token_mask = torch.tensor(
        [[True]],
        dtype=torch.bool,
    )
    positive_targets = torch.tensor(
        [[[True, False]]],
        dtype=torch.bool,
    )

    unweighted_loss = calculate_binary_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        temperature=1.0,
    )
    weighted_loss = calculate_binary_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        temperature=1.0,
        positive_targets=positive_targets,
        positive_weights=torch.tensor([3.0, 1.0]),
    )

    torch.testing.assert_close(
        weighted_loss,
        unweighted_loss * 2.0,
    )


def test_token_task_distillation_forwards_morphology_weights() -> None:
    teacher_logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2)),
        morphology_logits=(torch.tensor([[[2.0, 2.0]]]),),
        lemma_rule_logits=torch.zeros((1, 1, 2)),
    )
    student_logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2), requires_grad=True),
        morphology_logits=(
            torch.tensor(
                [[[-2.0, -2.0]]],
                requires_grad=True,
            ),
        ),
        lemma_rule_logits=torch.zeros((1, 1, 2), requires_grad=True),
    )
    token_mask = torch.tensor([[True]], dtype=torch.bool)
    morphology_targets = (torch.tensor([[[True, False]]]),)

    unweighted_losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        temperature=1.0,
    )
    weighted_losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        temperature=1.0,
        morphology_targets=morphology_targets,
        loss_weights=TokenTaskLossWeights(
            morphology_positive_weights=(torch.tensor([3.0, 1.0]),),
        ),
    )

    torch.testing.assert_close(
        weighted_losses.morphology_loss,
        unweighted_losses.morphology_loss * 2.0,
    )
    torch.testing.assert_close(
        weighted_losses.upos_loss,
        unweighted_losses.upos_loss,
    )
    torch.testing.assert_close(
        weighted_losses.lemma_rule_loss,
        unweighted_losses.lemma_rule_loss,
    )
