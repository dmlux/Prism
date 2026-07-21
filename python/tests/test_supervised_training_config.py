from prism.training import SupervisedTrainingConfig


def test_supervised_training_config_exposes_reproducible_policy() -> None:
    config = SupervisedTrainingConfig(
        epoch_count=3,
        batch_size=16,
        backbone_learning_rate=2e-5,
        task_head_learning_rate=5e-4,
        weight_decay=0.01,
        max_gradient_norm=1.0,
        warmup_ratio=0.1,
        random_seed=42,
        morphology_weight_cap=10.0,
    )

    assert config.epoch_count == 3
    assert config.batch_size == 16
    assert config.backbone_learning_rate == 2e-5
    assert config.task_head_learning_rate == 5e-4
    assert config.weight_decay == 0.01
    assert config.max_gradient_norm == 1.0
    assert config.warmup_ratio == 0.1
    assert config.random_seed == 42
    assert config.morphology_weight_cap == 10.0
