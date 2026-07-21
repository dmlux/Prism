import pytest

from prism.modeling import TokenPoolingStrategy, TokenTaskHeadArchitecture
from prism.training import (
    TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
)


def test_token_task_checkpoint_requires_hybrid_morphology_format() -> None:
    validate_token_task_checkpoint_format(
        {
            "checkpoint_format_version": TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
        }
    )

    with pytest.raises(
        ValueError,
        match="incompatible with the hybrid morphology contract",
    ):
        validate_token_task_checkpoint_format(
            {
                "checkpoint_format_version": 2,
            }
        )


def test_token_pooling_strategy_is_loaded_from_checkpoint_metadata() -> None:
    assert token_pooling_strategy_from_checkpoint({}) is TokenPoolingStrategy.FIRST
    assert (
        token_pooling_strategy_from_checkpoint({"token_pooling_strategy": "mean"})
        is TokenPoolingStrategy.MEAN
    )

    with pytest.raises(
        ValueError,
        match="Unsupported checkpoint token pooling strategy",
    ):
        token_pooling_strategy_from_checkpoint({"token_pooling_strategy": "maximum"})


def test_task_head_architecture_is_loaded_from_checkpoint_metadata() -> None:
    assert (
        token_task_head_architecture_from_checkpoint({})
        is TokenTaskHeadArchitecture.LINEAR
    )
    assert (
        token_task_head_architecture_from_checkpoint(
            {"token_task_head_architecture": "shared-mlp"}
        )
        is TokenTaskHeadArchitecture.SHARED_MLP
    )
    assert (
        token_task_head_architecture_from_checkpoint(
            {"token_task_head_architecture": "wide-shared-mlp"}
        )
        is TokenTaskHeadArchitecture.WIDE_SHARED_MLP
    )

    with pytest.raises(
        ValueError,
        match="Unsupported checkpoint task-head architecture",
    ):
        token_task_head_architecture_from_checkpoint(
            {"token_task_head_architecture": "separate-mlp"}
        )
