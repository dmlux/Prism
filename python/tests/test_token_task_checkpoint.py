import pytest
import torch

from prism.modeling import (
    BackboneLayerAggregationStrategy,
    TokenPoolingStrategy,
    TokenTaskHeadArchitecture,
)
from prism.training import (
    TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
    backbone_layer_aggregation_strategy_from_checkpoint,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
    character_vocabulary_from_checkpoint,
    maximum_character_count_from_checkpoint,
    morphology_logit_correction_from_checkpoint,
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
    assert (
        token_task_head_architecture_from_checkpoint(
            {"token_task_head_architecture": ("wide-shared-mlp-task-adapters")}
        )
        is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_TASK_ADAPTERS
    )
    assert (
        token_task_head_architecture_from_checkpoint(
            {"token_task_head_architecture": ("wide-shared-mlp-structured-morphology")}
        )
        is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY
    )
    assert (
        token_task_head_architecture_from_checkpoint(
            {
                "token_task_head_architecture": (
                    "wide-shared-mlp-structured-morphology-character-cnn"
                )
            }
        )
        is TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
    )

    with pytest.raises(
        ValueError,
        match="Unsupported checkpoint task-head architecture",
    ):
        token_task_head_architecture_from_checkpoint(
            {"token_task_head_architecture": "separate-mlp"}
        )


def test_backbone_layer_aggregation_is_loaded_from_checkpoint_metadata() -> None:
    assert (
        backbone_layer_aggregation_strategy_from_checkpoint({})
        is BackboneLayerAggregationStrategy.LAST
    )
    assert (
        backbone_layer_aggregation_strategy_from_checkpoint(
            {"backbone_layer_aggregation": "learned-last-four"}
        )
        is BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR
    )

    with pytest.raises(
        ValueError,
        match="Unsupported checkpoint backbone layer aggregation",
    ):
        backbone_layer_aggregation_strategy_from_checkpoint(
            {"backbone_layer_aggregation": "concatenate-all"}
        )


def test_character_contract_is_loaded_only_for_character_architecture() -> None:
    architecture = (
        TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
    )
    checkpoint = {
        "character_vocabulary": {
            "version": 1,
            "characters": ["a", "b"],
        },
        "maximum_character_count": 32,
    }

    vocabulary = character_vocabulary_from_checkpoint(
        checkpoint,
        architecture=architecture,
    )

    assert vocabulary is not None
    assert vocabulary.characters == ("a", "b")
    assert (
        maximum_character_count_from_checkpoint(
            checkpoint,
            architecture=architecture,
        )
        == 32
    )

    with pytest.raises(ValueError, match="must contain a character vocabulary"):
        character_vocabulary_from_checkpoint({}, architecture=architecture)


def test_morphology_logit_correction_uses_checkpointed_training_weights() -> None:
    checkpoint = {
        "morphology_weights": (
            (1.0, 4.0),
            (3.0,),
        )
    }

    assert (
        morphology_logit_correction_from_checkpoint(
            checkpoint,
            strength=0.0,
        )
        is None
    )

    correction = morphology_logit_correction_from_checkpoint(
        checkpoint,
        strength=0.5,
    )
    assert correction is not None
    assert correction.strength == 0.5
    torch.testing.assert_close(correction.weights[0], torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(correction.weights[1], torch.tensor([3.0]))

    with pytest.raises(ValueError, match="requires weights stored"):
        morphology_logit_correction_from_checkpoint({}, strength=0.5)
