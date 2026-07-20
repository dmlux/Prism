import math

import torch

from prism.data import TokenTargets
from prism.training import SupervisedTrainingConfig, TokenTaskLossWeights
from prism.training.class_weights import (
    build_token_task_loss_weights,
    calculate_binary_positive_weights,
    calculate_morphology_positive_weights,
)


def test_calculate_binary_positive_weights_ignores_padding() -> None:
    targets = torch.tensor(
        [
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [False, True, False],
                [False, True, True],
            ]
        ]
    )
    token_mask = torch.tensor([[True, True, True, True, False]])

    weights = calculate_binary_positive_weights(
        targets=targets,
        token_mask=token_mask,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([1.0, math.sqrt(3.0), 1.0]),
    )


def test_calculate_morphology_positive_weights_uses_all_tokens() -> None:
    targets = (
        TokenTargets(
            upos_id=0,
            morphology=((True, False, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((True, False, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((True, False, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((False, True, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
    )

    weights = calculate_morphology_positive_weights(
        targets=targets,
        maximum_weight=1.5,
    )

    assert len(weights) == 1
    torch.testing.assert_close(
        weights[0],
        torch.tensor([1.0, 1.5, 1.0]),
    )


def test_binary_positive_weights_respect_maximum_weight() -> None:
    targets = torch.tensor(
        [
            [
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [True, False],
                [False, True],
            ]
        ]
    )
    token_mask = torch.ones(
        (1, 10),
        dtype=torch.bool,
    )

    weights = calculate_binary_positive_weights(
        targets=targets,
        token_mask=token_mask,
        maximum_weight=2.0,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([1.0, 2.0]),
    )


def test_build_token_task_loss_weights_uses_training_config() -> None:
    config = SupervisedTrainingConfig(
        epoch_count=5,
        batch_size=16,
        backbone_learning_rate=2e-5,
        task_head_learning_rate=5e-4,
        weight_decay=0.01,
        max_gradient_norm=1.0,
        warmup_ratio=0.1,
        random_seed=42,
        morphology_positive_weight_cap=1.5,
    )
    targets = (
        TokenTargets(
            upos_id=0,
            morphology=((True, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((True, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((True, False),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((False, True),),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
    )

    loss_weights = build_token_task_loss_weights(
        targets=targets,
        config=config,
    )

    assert isinstance(loss_weights, TokenTaskLossWeights)
    torch.testing.assert_close(
        loss_weights.morphology_positive_weights[0],
        torch.tensor([1.0, 1.5]),
    )
