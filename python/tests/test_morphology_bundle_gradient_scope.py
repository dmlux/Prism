import pytest
import torch
from torch import nn

from prism.modeling import (
    MorphologyBundleCandidate,
    MorphologyBundleLossGradientScope,
    MorphologyBundleRerankerSpec,
    MorphologyPreHeadArchitecture,
    TokenTaskHeadArchitecture,
    TokenTaskHeads,
)
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)


def _schema() -> TokenTaskSchema:
    return TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Gender",
                    values=("Fem", "Masc"),
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


def _reranker_spec() -> MorphologyBundleRerankerSpec:
    return MorphologyBundleRerankerSpec(
        maximum_candidates_per_upos=2,
        candidates=(
            MorphologyBundleCandidate(
                upos_id=0,
                morphology=((False, True, False),),
                training_count=3,
            ),
            MorphologyBundleCandidate(
                upos_id=0,
                morphology=((False, False, True),),
                training_count=2,
            ),
            MorphologyBundleCandidate(
                upos_id=1,
                morphology=((True, False, False),),
                training_count=1,
            ),
        ),
    )


def _heads() -> TokenTaskHeads:
    return TokenTaskHeads(
        hidden_size=4,
        schema=_schema(),
        dropout_probability=0.2,
        architecture=(TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY),
        morphology_pre_head_architecture=(MorphologyPreHeadArchitecture.SHARED_MLP),
        morphology_bundle_reranker_spec=_reranker_spec(),
    )


def _has_gradient(module: nn.Module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


@pytest.mark.parametrize(
    (
        "scope",
        "expect_shared_gradient",
        "expect_upos_gradient",
        "expect_morphology_gradient",
        "expect_decoder_gradient",
    ),
    (
        (
            MorphologyBundleLossGradientScope.FULL,
            True,
            True,
            True,
            True,
        ),
        (
            MorphologyBundleLossGradientScope.MORPHOLOGY,
            False,
            False,
            True,
            True,
        ),
        (
            MorphologyBundleLossGradientScope.RESIDUAL_ONLY,
            False,
            False,
            False,
            False,
        ),
    ),
)
def test_bundle_loss_gradient_scope_updates_only_selected_parameters(
    scope: MorphologyBundleLossGradientScope,
    expect_shared_gradient: bool,
    expect_upos_gradient: bool,
    expect_morphology_gradient: bool,
    expect_decoder_gradient: bool,
) -> None:
    torch.manual_seed(7)
    heads = _heads()
    heads.set_morphology_bundle_loss_gradient_scope(scope)
    hidden_states = torch.randn((1, 2, 4), requires_grad=True)

    logits = heads(hidden_states)
    assert logits.morphology_bundle_scores is not None
    if scope is MorphologyBundleLossGradientScope.FULL:
        assert logits.morphology_bundle_loss_scores is None
        bundle_loss_scores = logits.morphology_bundle_scores
    else:
        assert logits.morphology_bundle_loss_scores is not None
        torch.testing.assert_close(
            logits.morphology_bundle_loss_scores,
            logits.morphology_bundle_scores,
        )
        bundle_loss_scores = logits.morphology_bundle_loss_scores

    bundle_loss_scores.square().sum().backward()

    assert (hidden_states.grad is not None) is expect_shared_gradient
    assert _has_gradient(heads.input_projection) is expect_shared_gradient
    assert _has_gradient(heads.upos_head) is expect_upos_gradient
    assert _has_gradient(heads.morphology_heads) is expect_morphology_gradient
    assert (
        _has_gradient(heads.morphology_pre_head_projection)
        is expect_morphology_gradient
    )
    assert heads.structured_morphology_decoder is not None
    assert _has_gradient(heads.structured_morphology_decoder) is expect_decoder_gradient
    assert not _has_gradient(heads.lemma_rule_head)
    assert heads.morphology_bundle_reranker is not None
    assert _has_gradient(heads.morphology_bundle_reranker.candidate_projection)
    assert heads.morphology_bundle_reranker.refinement_gates.grad is None
