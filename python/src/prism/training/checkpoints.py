from collections.abc import Mapping


TOKEN_TASK_CHECKPOINT_FORMAT_VERSION = 3


def validate_token_task_checkpoint_format(
    checkpoint: Mapping[str, object],
) -> None:
    format_version = checkpoint.get("checkpoint_format_version")

    if format_version != TOKEN_TASK_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Checkpoint format is incompatible with the hybrid morphology "
            f"contract: expected {TOKEN_TASK_CHECKPOINT_FORMAT_VERSION}, "
            f"got {format_version!r}."
        )
