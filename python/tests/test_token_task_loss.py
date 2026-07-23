import math

import torch

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenTaskLogits
from prism.schema import MorphologyFeatureSchema, MorphologySchema
from prism.training import (
    MorphologyBundleLossPolicy,
    TokenTaskLossWeights,
    calculate_morphology_bundle_loss,
    compute_token_task_loss,
)


def test_token_task_loss_combines_all_training_tasks() -> None:
    logits = TokenTaskLogits(
        upos_logits=torch.zeros(
            (1, 2, 2),
            requires_grad=True,
        ),
        morphology_logits=(
            torch.zeros((1, 2, 2), requires_grad=True),
            torch.zeros((1, 2, 2), requires_grad=True),
        ),
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
            torch.tensor(
                [[[False, True, False], [False, False, False]]],
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
        morphology_schema=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Categorical",
                    values=("Value",),
                    allows_multiple_values=False,
                ),
                MorphologyFeatureSchema(
                    name="MultiLabel",
                    values=("First", "Second"),
                    allows_multiple_values=True,
                ),
            ),
        ),
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
    assert logits.morphology_logits[1].grad is not None
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
        logits.morphology_logits[1].grad[0, 1],
        torch.zeros(2),
    )
    torch.testing.assert_close(
        logits.lemma_rule_logits.grad[0, 1],
        torch.zeros(2),
    )


def test_morphology_bundle_loss_marginalizes_matching_candidates() -> None:
    candidate_scores = torch.tensor(
        [[[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]]],
        requires_grad=True,
    )
    token_mask = torch.tensor([[True, True]], dtype=torch.bool)
    morphology_targets = (
        torch.tensor(
            [[[False, True], [True, False]]],
            dtype=torch.bool,
        ),
    )
    policy = MorphologyBundleLossPolicy(
        weight=1.0,
        candidate_morphology_targets=(
            torch.tensor(
                [
                    [False, True],
                    [True, False],
                    [False, True],
                ],
                dtype=torch.bool,
            ),
        ),
    )

    result = calculate_morphology_bundle_loss(
        candidate_scores=candidate_scores,
        morphology_targets=morphology_targets,
        token_mask=token_mask,
        policy=policy,
    )

    expected_first = -torch.logsumexp(
        torch.log_softmax(candidate_scores[0, 0], dim=-1)[[0, 2]],
        dim=0,
    )
    expected_second = -torch.log_softmax(
        candidate_scores[0, 1],
        dim=-1,
    )[1]
    torch.testing.assert_close(
        result.loss,
        (expected_first + expected_second) / 2,
    )
    assert result.target_count == 2
    assert result.token_count == 2
    assert result.coverage == 1.0

    result.loss.backward()
    assert candidate_scores.grad is not None


def test_token_task_loss_adds_weighted_complete_bundle_objective() -> None:
    bundle_scores = torch.zeros((1, 1, 2), requires_grad=True)
    logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2)),
        morphology_logits=(torch.zeros((1, 1, 2)),),
        lemma_rule_logits=torch.zeros((1, 1, 2)),
        morphology_bundle_scores=bundle_scores,
    )
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor([[0]]),
        morphology_targets=(torch.tensor([[[False, True]]], dtype=torch.bool),),
        lemma_rule_ids=torch.tensor([[0]]),
        lemma_rule_mask=torch.tensor([[True]]),
        token_mask=torch.tensor([[True]]),
    )

    losses = compute_token_task_loss(
        logits=logits,
        targets=targets,
        morphology_schema=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Feature",
                    values=("Value",),
                    allows_multiple_values=False,
                ),
            ),
        ),
        morphology_bundle_loss_policy=MorphologyBundleLossPolicy(
            weight=0.25,
            candidate_morphology_targets=(
                torch.tensor(
                    [[True, False], [False, True]],
                    dtype=torch.bool,
                ),
            ),
        ),
    )

    assert losses.morphology_bundle_loss is not None
    torch.testing.assert_close(
        losses.morphology_bundle_loss,
        torch.tensor(math.log(2.0)),
    )
    torch.testing.assert_close(
        losses.total_loss,
        torch.tensor(3.25 * math.log(2.0)),
    )
    assert losses.morphology_bundle_target_count == 1
    assert losses.morphology_bundle_token_count == 1

    losses.total_loss.backward()
    assert bundle_scores.grad is not None


def test_token_task_loss_applies_categorical_morphology_weights() -> None:
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
        morphology_weights=(torch.tensor([1.0, 3.0]),),
    )

    losses = compute_token_task_loss(
        logits=logits,
        targets=targets,
        morphology_schema=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Feature",
                    values=("Value",),
                    allows_multiple_values=False,
                ),
            ),
        ),
        loss_weights=loss_weights,
    )

    torch.testing.assert_close(
        losses.morphology_loss,
        torch.tensor(3.0 * math.log(2.0)),
    )


def test_token_task_loss_applies_multi_label_morphology_weights() -> None:
    logits = TokenTaskLogits(
        upos_logits=torch.zeros((1, 1, 2)),
        morphology_logits=(torch.zeros((1, 1, 2)),),
        lemma_rule_logits=torch.zeros((1, 1, 2)),
    )
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor([[0]]),
        morphology_targets=(
            torch.tensor(
                [[[False, True, False]]],
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor([[0]]),
        lemma_rule_mask=torch.tensor([[True]]),
        token_mask=torch.tensor([[True]]),
    )

    losses = compute_token_task_loss(
        logits=logits,
        targets=targets,
        morphology_schema=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Feature",
                    values=("First", "Second"),
                    allows_multiple_values=True,
                ),
            ),
        ),
        loss_weights=TokenTaskLossWeights(
            morphology_weights=(torch.tensor([3.0, 1.0]),),
        ),
    )

    torch.testing.assert_close(
        losses.morphology_loss,
        torch.tensor(2.0 * math.log(2.0)),
    )
