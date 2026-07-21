from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from prism.modeling import (
    BackboneLayerAggregationStrategy,
    CharacterCnnTokenEncoder,
    PretrainedBackboneSpec,
    SharedResidualTokenProjection,
    StructuredMorphologyDecoder,
    TaskResidualAdapter,
    TokenTaskHeadArchitecture,
    WideSharedResidualTokenProjection,
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

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ):
        wide_model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
            head_architecture=TokenTaskHeadArchitecture.WIDE_SHARED_MLP,
        )

    assert isinstance(
        wide_model.heads.input_projection,
        WideSharedResidualTokenProjection,
    )

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ):
        adapted_model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
            head_architecture=(TokenTaskHeadArchitecture.WIDE_SHARED_MLP_TASK_ADAPTERS),
        )

    assert isinstance(
        adapted_model.heads.input_projection,
        WideSharedResidualTokenProjection,
    )
    assert isinstance(adapted_model.heads.upos_adapter, TaskResidualAdapter)
    assert isinstance(
        adapted_model.heads.morphology_adapter,
        TaskResidualAdapter,
    )
    assert isinstance(adapted_model.heads.lemma_adapter, TaskResidualAdapter)

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ):
        structured_model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
            head_architecture=(
                TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY
            ),
        )

    assert isinstance(
        structured_model.heads.structured_morphology_decoder,
        StructuredMorphologyDecoder,
    )

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ):
        character_model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
            head_architecture=(
                TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
            ),
            character_vocabulary_size=32,
        )

    assert isinstance(character_model.character_encoder, CharacterCnnTokenEncoder)
    assert character_model.heads.character_fusion is not None

    with patch(
        "prism.modeling.taggers.load_backbone_model",
        return_value=backbone,
    ):
        mixed_model = build_pretrained_token_tagger(
            backbone_spec=spec,
            schema=schema,
            dropout_probability=0.1,
            layer_aggregation_strategy=(
                BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR
            ),
        )

    assert (
        mixed_model.layer_aggregation.strategy
        is BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR
    )
    assert (
        sum(
            parameter.numel()
            for parameter in mixed_model.layer_aggregation.parameters()
        )
        == 5
    )
