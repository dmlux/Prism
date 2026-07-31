from pathlib import Path

import pytest
import torch
from torch import nn

import prism.languages.norwegian.train_baseline as train_baseline
from prism.languages.norwegian import NORBERT4_BASE_BACKBONE
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)
from prism.schema.serialization import serialize_token_task_schema
from prism.training import TOKEN_TASK_CHECKPOINT_FORMAT_VERSION


def test_load_distillation_teacher_restores_frozen_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(
            version=1,
            labels=("NOUN",),
        ),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Sing",),
                    allows_multiple_values=False,
                ),
            ),
        ),
        lemma_rules=LemmaRuleSchema(
            version=1,
            rules=(
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=0,
                    prefix_addition="",
                    suffix_addition="",
                ),
            ),
        ),
    )
    fake_teacher = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "teacher.pt"

    torch.save(
        {
            "checkpoint_format_version": TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
            "language_tag": "no",
            "model_role": "teacher",
            "schema": serialize_token_task_schema(schema),
            "backbone_model_id": NORBERT4_BASE_BACKBONE.model_id,
            "backbone_revision": NORBERT4_BASE_BACKBONE.revision,
            "model_state_dict": fake_teacher.state_dict(),
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        train_baseline,
        "build_pretrained_token_tagger",
        lambda **_: fake_teacher,
    )

    from prism.languages.norwegian import NORWEGIAN_BOKMAAL_PROFILE

    loaded_teacher = train_baseline._load_distillation_teacher(
        checkpoint_path=checkpoint_path,
        profile=NORWEGIAN_BOKMAAL_PROFILE,
        schema=schema,
        requested_language_tag="nb",
        requested_treebank_release="current",
    )

    assert loaded_teacher is fake_teacher
    assert not loaded_teacher.training
    assert all(not parameter.requires_grad for parameter in loaded_teacher.parameters())
