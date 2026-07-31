import pytest
import torch

from prism.training import (
    RelationDistillationPolicy,
    compute_token_relation_loss,
)


def test_relation_policy_validates_configuration() -> None:
    policy = RelationDistillationPolicy(weight=1.0, relation_head_count=8)
    assert policy.weight == 1.0
    assert policy.relation_head_count == 8

    with pytest.raises(ValueError, match="weight must be positive"):
        RelationDistillationPolicy(weight=0.0)
    with pytest.raises(ValueError, match="head count must be positive"):
        RelationDistillationPolicy(weight=1.0, relation_head_count=0)


def test_relation_loss_is_zero_for_identical_relations() -> None:
    hidden = torch.tensor(
        [[[1.0, 2.0, -1.0, 0.5], [0.0, 1.0, 1.0, -0.5], [2.0, 0.0, 0.5, 1.0]]]
    )
    token_mask = torch.tensor([[True, True, True]])

    loss = compute_token_relation_loss(
        student_hidden_states=hidden.clone(),
        teacher_hidden_states=hidden.clone(),
        token_mask=token_mask,
        relation_head_count=2,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_relation_loss_supports_different_hidden_sizes() -> None:
    torch.manual_seed(7)
    student_hidden = torch.randn(2, 3, 4)
    teacher_hidden = torch.randn(2, 3, 8)
    token_mask = torch.tensor([[True, True, True], [True, True, False]])

    loss = compute_token_relation_loss(
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        token_mask=token_mask,
        relation_head_count=2,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_relation_loss_ignores_padding_tokens() -> None:
    torch.manual_seed(11)
    student_valid = torch.randn(1, 2, 4)
    teacher_valid = torch.randn(1, 2, 4)
    unpadded_loss = compute_token_relation_loss(
        student_hidden_states=student_valid,
        teacher_hidden_states=teacher_valid,
        token_mask=torch.tensor([[True, True]]),
        relation_head_count=2,
    )

    padding_garbage = torch.full((1, 1, 4), 123.0)
    padded_loss = compute_token_relation_loss(
        student_hidden_states=torch.cat((student_valid, padding_garbage), dim=1),
        teacher_hidden_states=torch.cat((teacher_valid, padding_garbage), dim=1),
        token_mask=torch.tensor([[True, True, False]]),
        relation_head_count=2,
    )

    torch.testing.assert_close(padded_loss, unpadded_loss)


def test_relation_loss_only_backpropagates_into_the_student() -> None:
    torch.manual_seed(13)
    student_hidden = torch.randn(1, 3, 4, requires_grad=True)
    teacher_hidden = torch.randn(1, 3, 8, requires_grad=True)

    loss = compute_token_relation_loss(
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        token_mask=torch.tensor([[True, True, True]]),
        relation_head_count=2,
    )
    loss.backward()

    assert student_hidden.grad is not None
    assert torch.any(student_hidden.grad != 0.0)
    assert teacher_hidden.grad is None


def test_relation_loss_validates_shapes() -> None:
    hidden = torch.zeros(1, 2, 4)
    token_mask = torch.tensor([[True, True]])

    with pytest.raises(ValueError, match="divisible"):
        compute_token_relation_loss(
            student_hidden_states=hidden,
            teacher_hidden_states=hidden,
            token_mask=token_mask,
            relation_head_count=3,
        )
    with pytest.raises(ValueError, match="same token sequence"):
        compute_token_relation_loss(
            student_hidden_states=hidden,
            teacher_hidden_states=torch.zeros(1, 3, 4),
            token_mask=token_mask,
            relation_head_count=2,
        )
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_token_relation_loss(
            student_hidden_states=hidden,
            teacher_hidden_states=hidden,
            token_mask=torch.tensor([[False, False]]),
            relation_head_count=2,
        )
