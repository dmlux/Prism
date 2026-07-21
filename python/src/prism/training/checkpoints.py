from collections.abc import Mapping

from prism.modeling import TokenPoolingStrategy


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


def token_pooling_strategy_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> TokenPoolingStrategy:
    raw_strategy = checkpoint.get(
        "token_pooling_strategy",
        TokenPoolingStrategy.FIRST.value,
    )
    if not isinstance(raw_strategy, str):
        raise ValueError("Checkpoint token pooling strategy must be a string.")

    try:
        return TokenPoolingStrategy(raw_strategy)
    except ValueError as error:
        raise ValueError(
            f"Unsupported checkpoint token pooling strategy: {raw_strategy!r}."
        ) from error
