from collections.abc import Mapping

import torch

from prism.modeling import (
    BackboneLayerAggregationStrategy,
    MorphologyLogitCorrection,
    MorphologyBundleRerankerSpec,
    MorphologyAgreementRefinerSpec,
    TokenPoolingStrategy,
    TokenTaskHeadArchitecture,
)
from prism.schema import CharacterVocabularySchema
from prism.schema.serialization import deserialize_character_vocabulary_schema
from prism.training.morphology_bundle_reranking import (
    deserialize_morphology_bundle_reranker_spec,
)
from prism.training.morphology_agreement import (
    deserialize_morphology_agreement_refiner_spec,
)


TOKEN_TASK_CHECKPOINT_FORMAT_VERSION = 3


def morphology_agreement_refiner_spec_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> MorphologyAgreementRefinerSpec | None:
    raw_spec = checkpoint.get("morphology_agreement_refiner")
    if raw_spec is None:
        return None
    return deserialize_morphology_agreement_refiner_spec(raw_spec)


def morphology_bundle_reranker_spec_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> MorphologyBundleRerankerSpec | None:
    raw_spec = checkpoint.get("morphology_bundle_reranker")
    if raw_spec is None:
        return None
    return deserialize_morphology_bundle_reranker_spec(raw_spec)


def morphology_logit_correction_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    strength: float,
) -> MorphologyLogitCorrection | None:
    """Build an optional evaluation correction from checkpointed loss weights."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "Morphology logit-correction strength must be between zero and one."
        )
    if strength == 0.0:
        return None

    raw_weights = checkpoint.get("morphology_weights")
    if not isinstance(raw_weights, (list, tuple)) or not raw_weights:
        raise ValueError(
            "Morphology logit correction requires weights stored in the checkpoint."
        )

    weights: list[torch.Tensor] = []
    for feature_weights in raw_weights:
        if not isinstance(feature_weights, (list, tuple)) or not feature_weights:
            raise ValueError("Checkpoint morphology weights are invalid.")
        if any(
            isinstance(weight, bool) or not isinstance(weight, (int, float))
            for weight in feature_weights
        ):
            raise ValueError("Checkpoint morphology weights must be numeric.")
        weights.append(torch.tensor(feature_weights, dtype=torch.float32))

    return MorphologyLogitCorrection(
        strength=strength,
        weights=tuple(weights),
    )


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


def token_task_head_architecture_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> TokenTaskHeadArchitecture:
    raw_architecture = checkpoint.get(
        "token_task_head_architecture",
        TokenTaskHeadArchitecture.LINEAR.value,
    )
    if not isinstance(raw_architecture, str):
        raise ValueError("Checkpoint task-head architecture must be a string.")

    try:
        return TokenTaskHeadArchitecture(raw_architecture)
    except ValueError as error:
        raise ValueError(
            f"Unsupported checkpoint task-head architecture: {raw_architecture!r}."
        ) from error


def backbone_layer_aggregation_strategy_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> BackboneLayerAggregationStrategy:
    raw_strategy = checkpoint.get(
        "backbone_layer_aggregation",
        BackboneLayerAggregationStrategy.LAST.value,
    )
    if not isinstance(raw_strategy, str):
        raise ValueError("Checkpoint backbone layer aggregation must be a string.")

    try:
        return BackboneLayerAggregationStrategy(raw_strategy)
    except ValueError as error:
        raise ValueError(
            f"Unsupported checkpoint backbone layer aggregation: {raw_strategy!r}."
        ) from error


def character_vocabulary_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    architecture: TokenTaskHeadArchitecture,
) -> CharacterVocabularySchema | None:
    raw_vocabulary = checkpoint.get("character_vocabulary")
    if architecture.uses_character_encoder:
        if raw_vocabulary is None:
            raise ValueError(
                "Character-aware checkpoint must contain a character vocabulary."
            )
        return deserialize_character_vocabulary_schema(raw_vocabulary)

    if raw_vocabulary is not None:
        raise ValueError(
            "Non-character checkpoint must not contain a character vocabulary."
        )

    return None


def maximum_character_count_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    architecture: TokenTaskHeadArchitecture,
) -> int | None:
    raw_count = checkpoint.get("maximum_character_count")
    if architecture.uses_character_encoder:
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count < 5
        ):
            raise ValueError(
                "Character-aware checkpoint maximum character count must be "
                "an integer of at least five."
            )
        return raw_count

    if raw_count is not None:
        raise ValueError(
            "Non-character checkpoint must not define a maximum character count."
        )

    return None
