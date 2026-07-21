from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from prism.modeling import (
    PretrainedBackboneSpec,
    SharedResidualTokenProjection,
    TokenTaskHeadArchitecture,
    build_pretrained_token_tagger,
)
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)


class ConfiguredBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)


def test_build_pretrained_token_tagger_uses_backbone_hidden_size() -> None:
    spec = PretrainedBackboneSpec(
        model_id="organization/model",
        revision="0123456789abcdef0123456789abcdef01234567",
        trust_remote_code=False,
    )
    schema = TokenTaskSchema(
        upos=UposSchema(
            version=1,
            labels=("NOUN", "VERB"),
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
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=1,
                    prefix_addition="",
                    suffix_addition="",
                ),
            ),
        ),
    )
    backbone = ConfiguredBackbone()

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ) as load_backbone:
        model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
        )

    load_backbone.assert_called_once_with(spec)
    assert model.backbone is backbone
    assert model.heads.upos_head.projection.in_features == 4
    assert model.heads.upos_head.projection.out_features == 2

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ):
        nonlinear_model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
            head_architecture=TokenTaskHeadArchitecture.SHARED_MLP,
        )

    assert isinstance(
        nonlinear_model.heads.input_projection,
        SharedResidualTokenProjection,
    )
