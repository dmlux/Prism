import torch
from torch import nn

from prism.modeling import (
    SharedResidualTokenProjection,
    StructuredMorphologyDecoder,
    TaskResidualAdapter,
    TokenTaskHeadArchitecture,
    TokenTaskHeads,
    WideSharedResidualTokenProjection,
)
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)


def test_token_task_heads_create_logits_from_schema() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(
            version=1,
            labels=("ADJ", "NOUN", "VERB"),
        ),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
                ),
                MorphologyFeatureSchema(
                    name="Tense",
                    values=("Past", "Pres"),
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
    heads = TokenTaskHeads(
        hidden_size=192,
        schema=schema,
        dropout_probability=0.1,
    )

    heads.eval()
    hidden_states = torch.randn((2, 4, 192))

    logits = heads(hidden_states)
    rescaled_logits = heads(hidden_states * 10.0 + 50.0)

    assert logits.upos_logits.shape == (2, 4, 3)
    assert tuple(
        feature_logits.shape for feature_logits in logits.morphology_logits
    ) == (
        (2, 4, 2),
        (2, 4, 3),
    )
    assert logits.lemma_rule_logits.shape == (2, 4, 2)

    torch.testing.assert_close(
        logits.upos_logits,
        rescaled_logits.upos_logits,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        logits.lemma_rule_logits,
        rescaled_logits.lemma_rule_logits,
        rtol=1e-4,
        atol=1e-5,
    )

    for logits_for_feature, rescaled_for_feature in zip(
        logits.morphology_logits,
        rescaled_logits.morphology_logits,
        strict=True,
    ):
        torch.testing.assert_close(
            logits_for_feature,
            rescaled_for_feature,
            rtol=1e-4,
            atol=1e-5,
        )


def test_token_task_heads_select_shared_residual_mlp() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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

    linear_heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
    )
    nonlinear_heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
        architecture=TokenTaskHeadArchitecture.SHARED_MLP,
    )

    assert isinstance(linear_heads.input_projection, nn.Identity)
    assert isinstance(
        nonlinear_heads.input_projection,
        SharedResidualTokenProjection,
    )
    assert (
        sum(
            parameter.numel()
            for parameter in nonlinear_heads.input_projection.parameters()
        )
        == 20
    )

    logits = nonlinear_heads(torch.randn((2, 3, 4)))

    assert logits.upos_logits.shape == (2, 3, 2)
    assert logits.morphology_logits[0].shape == (2, 3, 2)
    assert logits.lemma_rule_logits.shape == (2, 3, 2)


def test_token_task_heads_select_wide_shared_residual_mlp() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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

    heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
        architecture=TokenTaskHeadArchitecture.WIDE_SHARED_MLP,
    )

    assert isinstance(
        heads.input_projection,
        WideSharedResidualTokenProjection,
    )
    assert heads.input_projection.input_projection.in_features == 4
    assert heads.input_projection.input_projection.out_features == 8
    assert heads.input_projection.output_projection.in_features == 8
    assert heads.input_projection.output_projection.out_features == 4
    assert (
        sum(parameter.numel() for parameter in heads.input_projection.parameters())
        == 76
    )

    logits = heads(torch.randn((2, 3, 4)))

    assert logits.upos_logits.shape == (2, 3, 2)
    assert logits.morphology_logits[0].shape == (2, 3, 2)
    assert logits.lemma_rule_logits.shape == (2, 3, 2)


def test_token_task_heads_add_separate_task_residual_adapters() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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
    heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
        architecture=(TokenTaskHeadArchitecture.WIDE_SHARED_MLP_TASK_ADAPTERS),
    )

    assert isinstance(
        heads.input_projection,
        WideSharedResidualTokenProjection,
    )
    assert isinstance(heads.upos_adapter, TaskResidualAdapter)
    assert isinstance(heads.morphology_adapter, TaskResidualAdapter)
    assert isinstance(heads.lemma_adapter, TaskResidualAdapter)
    assert heads.upos_adapter is not heads.morphology_adapter
    assert heads.morphology_adapter is not heads.lemma_adapter

    for adapter in (
        heads.upos_adapter,
        heads.morphology_adapter,
        heads.lemma_adapter,
    ):
        assert adapter.input_projection.in_features == 4
        assert adapter.input_projection.out_features == 2
        assert adapter.output_projection.in_features == 2
        assert adapter.output_projection.out_features == 4
        assert sum(parameter.numel() for parameter in adapter.parameters()) == 22

    hidden_states = torch.randn((2, 3, 4))
    torch.testing.assert_close(
        heads.upos_adapter(hidden_states),
        hidden_states,
    )
    exported_adapter = torch.export.export(
        heads.upos_adapter,
        (hidden_states,),
        strict=True,
    )
    torch.testing.assert_close(
        exported_adapter.module()(hidden_states),
        hidden_states,
    )

    logits = heads(hidden_states)

    assert logits.upos_logits.shape == (2, 3, 2)
    assert logits.morphology_logits[0].shape == (2, 3, 2)
    assert logits.lemma_rule_logits.shape == (2, 3, 2)


def test_token_task_heads_select_structured_morphology_decoder() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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
    heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
        architecture=(TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY),
    )

    assert isinstance(
        heads.input_projection,
        WideSharedResidualTokenProjection,
    )
    assert isinstance(heads.upos_adapter, nn.Identity)
    assert isinstance(heads.morphology_adapter, nn.Identity)
    assert isinstance(heads.lemma_adapter, nn.Identity)
    assert isinstance(
        heads.structured_morphology_decoder,
        StructuredMorphologyDecoder,
    )

    logits = heads(torch.randn((2, 3, 4)))

    assert logits.upos_logits.shape == (2, 3, 2)
    assert logits.morphology_logits[0].shape == (2, 3, 2)
    assert logits.lemma_rule_logits.shape == (2, 3, 2)


def test_character_architecture_keeps_upos_independent_of_characters() -> None:
    torch.manual_seed(7)
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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
    heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
        architecture=(
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
        ),
    )
    contextual = torch.randn(1, 2, 4)

    first = heads(
        contextual,
        character_hidden_states=torch.tensor(
            [[[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]]
        ),
    )
    second = heads(
        contextual,
        character_hidden_states=torch.tensor(
            [[[3.0, 1.0, 0.0, 2.0], [1.0, 3.0, 2.0, 0.0]]]
        ),
    )

    torch.testing.assert_close(first.upos_logits, second.upos_logits)
    assert not torch.equal(first.lemma_rule_logits, second.lemma_rule_logits)
    assert not torch.equal(
        first.morphology_logits[0],
        second.morphology_logits[0],
    )
