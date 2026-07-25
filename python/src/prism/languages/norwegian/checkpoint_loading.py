"""Strict, reusable loading of Norwegian token-tagger checkpoints.

Evaluation, calibration, and silver labeling all need the same steps: load a
checkpoint, verify its format, treebank release, language support, schema,
and backbone identity against the pinned data, rebuild the exact architecture
from the stored metadata, and restore the weights strictly. This module owns
that sequence once so the command-line entry points stay thin.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from prism.conllu import read_sentences
from prism.data import build_norwegian_schema
from prism.languages import ModelRole
from prism.languages.norwegian.profile import (
    norwegian_model_supports_language_tag,
    norwegian_profile_for_language_tag,
)
from prism.modeling import (
    build_pretrained_token_tagger,
    load_backbone_tokenizer,
)
from prism.schema import CharacterVocabularySchema, TokenTaskSchema
from prism.schema.serialization import serialize_token_task_schema
from prism.training import (
    backbone_layer_aggregation_strategy_from_checkpoint,
    character_vocabulary_from_checkpoint,
    maximum_character_count_from_checkpoint,
    morphology_pre_head_architecture_from_checkpoint,
    morphology_bundle_reranker_spec_from_checkpoint,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedNorwegianTagger:
    checkpoint: dict
    model: nn.Module
    schema: TokenTaskSchema
    tokenizer: PreTrainedTokenizerBase
    character_vocabulary: CharacterVocabularySchema | None
    maximum_character_count: int
    batch_size: int

    @property
    def epoch_index(self) -> int:
        return int(self.checkpoint["epoch_index"])


def load_norwegian_token_tagger(
    *,
    checkpoint_path: Path,
    required_language_tags: tuple[str, ...],
    treebank_release: str,
) -> LoadedNorwegianTagger:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_token_task_checkpoint_format(checkpoint)

    checkpoint_treebank_release = checkpoint.get("treebank_release", "current")
    if checkpoint_treebank_release != treebank_release:
        raise ValueError(
            "Checkpoint treebank release does not match the requested release: "
            f"{checkpoint_treebank_release!r}"
        )
    checkpoint_language_tag = checkpoint.get("language_tag")
    if not isinstance(checkpoint_language_tag, str) or any(
        not norwegian_model_supports_language_tag(
            checkpoint_language_tag,
            language_tag,
        )
        for language_tag in required_language_tags
    ):
        raise ValueError(
            "Checkpoint language tag does not support the selected profiles: "
            f"{checkpoint_language_tag!r}"
        )

    raw_model_role = checkpoint.get("model_role", "student")
    if raw_model_role not in ("student", "teacher"):
        raise ValueError(f"Checkpoint model role is invalid: {raw_model_role!r}")
    model_role = cast(ModelRole, raw_model_role)

    raw_schema_language_tags = checkpoint.get("schema_language_tags")
    if raw_schema_language_tags is None:
        schema_language_tags = required_language_tags
    elif isinstance(raw_schema_language_tags, (list, tuple)) and all(
        isinstance(language_tag, str) for language_tag in raw_schema_language_tags
    ):
        schema_language_tags = tuple(raw_schema_language_tags)
    else:
        raise ValueError("Checkpoint schema language tags are invalid.")

    schema_profiles = tuple(
        norwegian_profile_for_language_tag(
            language_tag,
            treebank_release=treebank_release,
        )
        for language_tag in schema_language_tags
    )
    schema_training_tokens = tuple(
        sentence
        for schema_profile in schema_profiles
        for sentence in read_sentences(schema_profile.gold_treebank.training_path)
    )
    schema = build_norwegian_schema(schema_training_tokens)
    if checkpoint["schema"] != serialize_token_task_schema(schema):
        raise ValueError("Checkpoint schema does not match the pinned training data.")

    backbone_spec = schema_profiles[0].backbone_for_role(model_role)
    if checkpoint["backbone_model_id"] != backbone_spec.model_id:
        raise ValueError("Checkpoint backbone model does not match.")
    if checkpoint["backbone_revision"] != backbone_spec.revision:
        raise ValueError("Checkpoint backbone revision does not match.")

    tokenizer = load_backbone_tokenizer(backbone_spec)
    head_architecture = token_task_head_architecture_from_checkpoint(checkpoint)
    character_vocabulary = character_vocabulary_from_checkpoint(
        checkpoint,
        architecture=head_architecture,
    )
    maximum_character_count = maximum_character_count_from_checkpoint(
        checkpoint,
        architecture=head_architecture,
    )
    model = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
        pooling_strategy=token_pooling_strategy_from_checkpoint(checkpoint),
        head_architecture=head_architecture,
        morphology_pre_head_architecture=(
            morphology_pre_head_architecture_from_checkpoint(checkpoint)
        ),
        layer_aggregation_strategy=(
            backbone_layer_aggregation_strategy_from_checkpoint(checkpoint)
        ),
        character_vocabulary_size=(
            None if character_vocabulary is None else character_vocabulary.size
        ),
        morphology_bundle_reranker_spec=(
            morphology_bundle_reranker_spec_from_checkpoint(checkpoint)
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return LoadedNorwegianTagger(
        checkpoint=checkpoint,
        model=model,
        schema=schema,
        tokenizer=tokenizer,
        character_vocabulary=character_vocabulary,
        maximum_character_count=(
            32 if maximum_character_count is None else maximum_character_count
        ),
        batch_size=int(checkpoint["training_config"]["batch_size"]),
    )
