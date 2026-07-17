from torch import nn

from prism.training import (
    SupervisedTrainingConfig,
    build_supervised_adamw_optimizer,
)


def test_optimizer_uses_separate_backbone_and_head_policies() -> None:
    backbone = nn.Linear(4, 4)
    task_heads = nn.Linear(4, 2)
    config = SupervisedTrainingConfig(
        epoch_count=3,
        batch_size=16,
        backbone_learning_rate=2e-5,
        task_head_learning_rate=5e-4,
        weight_decay=0.01,
        max_gradient_norm=1.0,
        warmup_ratio=0.1,
        random_seed=42,
    )

    optimizer = build_supervised_adamw_optimizer(
        backbone=backbone,
        task_heads=task_heads,
        config=config,
    )

    groups_by_name = {group["name"]: group for group in optimizer.param_groups}

    assert groups_by_name["backbone_decay"]["lr"] == 2e-5
    assert groups_by_name["backbone_decay"]["weight_decay"] == 0.01
    assert groups_by_name["backbone_no_decay"]["lr"] == 2e-5
    assert groups_by_name["backbone_no_decay"]["weight_decay"] == 0.0

    assert groups_by_name["task_heads_decay"]["lr"] == 5e-4
    assert groups_by_name["task_heads_decay"]["weight_decay"] == 0.01
    assert groups_by_name["task_heads_no_decay"]["lr"] == 5e-4
    assert groups_by_name["task_heads_no_decay"]["weight_decay"] == 0.0

    assert any(
        parameter is backbone.weight
        for parameter in groups_by_name["backbone_decay"]["params"]
    )
    assert any(
        parameter is backbone.bias
        for parameter in groups_by_name["backbone_no_decay"]["params"]
    )
    assert any(
        parameter is task_heads.weight
        for parameter in groups_by_name["task_heads_decay"]["params"]
    )
    assert any(
        parameter is task_heads.bias
        for parameter in groups_by_name["task_heads_no_decay"]["params"]
    )
