import pytest
import torch

from prism.modeling import TokenTaskLogits
from prism.schema import MorphologyFeatureSchema, MorphologySchema
from prism.training import (
    TokenTaskDistillationPolicy,
    TokenTaskLossWeights,
    TokenTaskLosses,
    calculate_binary_distillation_loss,
    calculate_categorical_distillation_loss,
    calculate_decoupled_categorical_distillation_loss,
    combine_token_task_losses,
    compute_token_task_distillation_loss,
)


CATEGORICAL_MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Feature",
            values=("Value",),
            allows_multiple_values=False,
        ),
    ),
)
MULTI_LABEL_MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Feature",
            values=("First", "Second"),
            allows_multiple_values=True,
        ),
    ),
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


def test_decoupled_distillation_is_zero_for_identical_logits() -> None:
    logits = torch.tensor(
        [[[2.0, 0.5, -1.0], [10.0, -10.0, 0.0]]],
        requires_grad=True,
    )
    token_mask = torch.tensor([[True, False]], dtype=torch.bool)
    target_ids = torch.tensor([[0, 1]], dtype=torch.long)

    loss = calculate_decoupled_categorical_distillation_loss(
        student_logits=logits,
        teacher_logits=logits.detach(),
        target_ids=target_ids,
        token_mask=token_mask,
        temperature=2.0,
        target_class_weight=1.0,
        non_target_class_weight=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_decoupled_distillation_separates_target_and_non_target_knowledge() -> None:
    teacher_logits = torch.tensor([[[2.0, 1.0, -1.0]]])
    student_logits = torch.tensor(
        [[[2.0, -1.0, 1.0]]],
        requires_grad=True,
    )
    token_mask = torch.tensor([[True]], dtype=torch.bool)
    target_ids = torch.tensor([[0]], dtype=torch.long)

    target_only_loss = calculate_decoupled_categorical_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_ids=target_ids,
        token_mask=token_mask,
        temperature=1.0,
        target_class_weight=1.0,
        non_target_class_weight=0.0,
    )
    non_target_only_loss = calculate_decoupled_categorical_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_ids=target_ids,
        token_mask=token_mask,
        temperature=1.0,
        target_class_weight=0.0,
        non_target_class_weight=1.0,
    )

    torch.testing.assert_close(
        target_only_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    assert non_target_only_loss.item() > 0.0


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
        policy=TokenTaskDistillationPolicy.uniform(
            temperature=2.0,
            weight=0.5,
        ),
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
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
        policy=TokenTaskDistillationPolicy.uniform(
            temperature=2.0,
            weight=0.5,
        ),
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
        policy=TokenTaskDistillationPolicy.uniform(
            temperature=2.0,
            weight=0.5,
        ),
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
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


def test_categorical_distillation_weights_target_class() -> None:
    teacher_logits = torch.tensor([[[2.0, -2.0]]])
    student_logits = torch.tensor(
        [[[-2.0, 2.0]]],
        requires_grad=True,
    )
    token_mask = torch.tensor([[True]], dtype=torch.bool)

    unweighted_loss = calculate_categorical_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        temperature=1.0,
    )
    weighted_loss = calculate_categorical_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        temperature=1.0,
        target_ids=torch.tensor([[1]], dtype=torch.long),
        class_weights=torch.tensor([1.0, 3.0]),
    )

    torch.testing.assert_close(
        weighted_loss,
        unweighted_loss * 3.0,
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
    morphology_targets = (
        torch.tensor(
            [[[False, True, False]]],
            dtype=torch.bool,
        ),
    )

    unweighted_losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        policy=TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        ),
        morphology_schema=MULTI_LABEL_MORPHOLOGY_SCHEMA,
    )
    weighted_losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        policy=TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        ),
        morphology_schema=MULTI_LABEL_MORPHOLOGY_SCHEMA,
        morphology_targets=morphology_targets,
        loss_weights=TokenTaskLossWeights(
            morphology_weights=(torch.tensor([3.0, 1.0]),),
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


def test_combined_loss_applies_task_specific_weights() -> None:
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
    policy = TokenTaskDistillationPolicy(
        upos_temperature=1.0,
        morphology_temperature=1.0,
        lemma_rule_temperature=1.0,
        upos_weight=0.1,
        morphology_weight=0.2,
        lemma_rule_weight=0.3,
    )

    losses = combine_token_task_losses(
        supervised_losses=supervised_losses,
        distillation_losses=distillation_losses,
        policy=policy,
    )

    torch.testing.assert_close(losses.total_loss, torch.tensor(6.35))


def test_token_task_distillation_applies_task_specific_temperatures() -> None:
    teacher_logits = TokenTaskLogits(
        upos_logits=torch.tensor([[[2.0, -1.0]]]),
        morphology_logits=(torch.tensor([[[1.0, -2.0]]]),),
        lemma_rule_logits=torch.tensor([[[-1.0, 2.0]]]),
    )
    student_logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2), requires_grad=True),
        morphology_logits=(torch.zeros((1, 1, 2), requires_grad=True),),
        lemma_rule_logits=torch.zeros((1, 1, 2), requires_grad=True),
    )
    token_mask = torch.tensor([[True]], dtype=torch.bool)
    policy = TokenTaskDistillationPolicy(
        upos_temperature=1.0,
        morphology_temperature=2.0,
        lemma_rule_temperature=3.0,
        upos_weight=0.1,
        morphology_weight=0.2,
        lemma_rule_weight=0.3,
    )

    losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        policy=policy,
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
    )

    torch.testing.assert_close(
        losses.upos_loss,
        calculate_categorical_distillation_loss(
            student_logits=student_logits.upos_logits,
            teacher_logits=teacher_logits.upos_logits,
            token_mask=token_mask,
            temperature=1.0,
        ),
    )
    torch.testing.assert_close(
        losses.morphology_loss,
        calculate_categorical_distillation_loss(
            student_logits=student_logits.morphology_logits[0],
            teacher_logits=teacher_logits.morphology_logits[0],
            token_mask=token_mask,
            temperature=2.0,
        ),
    )
    torch.testing.assert_close(
        losses.lemma_rule_loss,
        calculate_categorical_distillation_loss(
            student_logits=student_logits.lemma_rule_logits,
            teacher_logits=teacher_logits.lemma_rule_logits,
            token_mask=token_mask,
            temperature=3.0,
        ),
    )


def test_token_task_distillation_uses_dkd_for_categorical_tasks() -> None:
    teacher_logits = TokenTaskLogits(
        upos_logits=torch.tensor([[[2.0, 1.0, -1.0]]]),
        morphology_logits=(torch.tensor([[[2.0, 1.0]]]),),
        lemma_rule_logits=torch.tensor([[[2.0, -1.0, 1.0]]]),
    )
    student_logits = TokenTaskLogits(
        upos_logits=torch.tensor([[[2.0, -1.0, 1.0]]], requires_grad=True),
        morphology_logits=(torch.tensor([[[2.0, 1.0]]], requires_grad=True),),
        lemma_rule_logits=torch.tensor(
            [[[2.0, 1.0, -1.0]]],
            requires_grad=True,
        ),
    )
    token_mask = torch.tensor([[True]], dtype=torch.bool)
    policy = TokenTaskDistillationPolicy.uniform(
        temperature=1.0,
        weight=0.1,
        categorical_objective="dkd",
        target_class_weight=0.0,
        non_target_class_weight=1.0,
    )

    losses = compute_token_task_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        token_mask=token_mask,
        lemma_rule_mask=token_mask,
        policy=policy,
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
        upos_ids=torch.tensor([[0]], dtype=torch.long),
        morphology_targets=(torch.tensor([[[True, False]]], dtype=torch.bool),),
        lemma_rule_ids=torch.tensor([[0]], dtype=torch.long),
    )

    assert losses.upos_loss.item() > 0.0
    torch.testing.assert_close(
        losses.morphology_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    assert losses.lemma_rule_loss.item() > 0.0


def test_distillation_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="temperatures"):
        TokenTaskDistillationPolicy.uniform(
            temperature=0.0,
            weight=0.1,
        )

    with pytest.raises(ValueError, match="weights"):
        TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=-0.1,
        )

    with pytest.raises(ValueError, match="objective"):
        TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
            categorical_objective="unknown",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="DKD component weights"):
        TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
            categorical_objective="dkd",
            non_target_class_weight=-1.0,
        )
