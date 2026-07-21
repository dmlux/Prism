import pytest

from prism.training import (
    TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
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
