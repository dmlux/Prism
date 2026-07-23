import torch

from prism.data import TokenTargets
from prism.modeling import (
    MorphologyBundleCandidate,
    MorphologyBundleReranker,
    MorphologyBundleRerankerSpec,
)
from prism.schema import MorphologyFeatureSchema, MorphologySchema
from prism.training import (
    build_morphology_bundle_reranker_spec,
    deserialize_morphology_bundle_reranker_spec,
    serialize_morphology_bundle_reranker_spec,
)


def _schema() -> MorphologySchema:
    return MorphologySchema(
        version=1,
        features=(
            MorphologyFeatureSchema(
                name="Case",
                values=("Acc", "Nom"),
                allows_multiple_values=True,
            ),
            MorphologyFeatureSchema(
                name="Number",
                values=("Plur", "Sing"),
                allows_multiple_values=False,
            ),
        ),
    )


def _spec() -> MorphologyBundleRerankerSpec:
    return MorphologyBundleRerankerSpec(
        maximum_candidates_per_upos=2,
        candidates=(
            MorphologyBundleCandidate(
                upos_id=0,
                morphology=((False, False, True), (False, False, True)),
                training_count=3,
            ),
            MorphologyBundleCandidate(
                upos_id=0,
                morphology=((False, True, False), (False, True, False)),
                training_count=1,
            ),
            MorphologyBundleCandidate(
                upos_id=1,
                morphology=((True, False, False), (True, False, False)),
                training_count=2,
            ),
        ),
    )


def test_bundle_reranker_is_residual_trainable_and_disableable() -> None:
    reranker = MorphologyBundleReranker(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=_schema(),
        spec=_spec(),
        dropout_probability=0.0,
    )
    hidden_states = torch.randn((1, 2, 4))
    upos_logits = torch.randn((1, 2, 2))
    morphology_logits = (
        torch.randn((1, 2, 2), requires_grad=True),
        torch.randn((1, 2, 3), requires_grad=True),
    )

    refined = reranker(
        hidden_states=hidden_states,
        upos_logits=upos_logits,
        morphology_logits=morphology_logits,
    )
    assert refined[0].shape == morphology_logits[0].shape
    assert refined[1].shape == morphology_logits[1].shape
    assert not torch.equal(refined[0], morphology_logits[0])

    sum(tensor.sum() for tensor in refined).backward()
    assert reranker.candidate_projection.weight.grad is not None

    reranker.set_enabled(False)
    disabled = reranker(
        hidden_states=hidden_states,
        upos_logits=upos_logits,
        morphology_logits=morphology_logits,
    )
    for actual, expected in zip(disabled, morphology_logits, strict=True):
        assert actual is expected


def test_bundle_reranker_can_isolate_direct_loss_gradient() -> None:
    reranker = MorphologyBundleReranker(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=_schema(),
        spec=_spec(),
        dropout_probability=0.0,
    )
    reranker.set_direct_loss_gradient_isolation(True)
    hidden_states = torch.randn((1, 2, 4), requires_grad=True)
    upos_logits = torch.randn((1, 2, 2), requires_grad=True)
    morphology_logits = (
        torch.randn((1, 2, 2), requires_grad=True),
        torch.randn((1, 2, 3), requires_grad=True),
    )

    _, candidate_scores, isolated_loss_scores = reranker.refine_with_training_scores(
        hidden_states=hidden_states,
        upos_logits=upos_logits,
        morphology_logits=morphology_logits,
    )

    assert candidate_scores is not None
    assert isolated_loss_scores is not None
    torch.testing.assert_close(isolated_loss_scores, candidate_scores)

    isolated_loss_scores.square().sum().backward()

    assert hidden_states.grad is None
    assert upos_logits.grad is None
    assert all(logits.grad is None for logits in morphology_logits)
    assert reranker.candidate_projection.weight.grad is not None
    assert reranker.candidate_projection.bias.grad is not None
    assert reranker.refinement_gates.grad is None


def test_bundle_reranker_spec_uses_top_k_and_round_trips() -> None:
    targets = (
        TokenTargets(
            upos_id=0,
            morphology=((False, True, False), (False, False, True)),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((False, True, False), (False, False, True)),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((False, False, True), (False, True, False)),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
        TokenTargets(
            upos_id=0,
            morphology=((True, False, False), (True, False, False)),
            lemma_is_annotated=False,
            lemma_rule_id=None,
        ),
    )

    spec = build_morphology_bundle_reranker_spec(
        targets=targets,
        maximum_candidates_per_upos=2,
    )

    assert len(spec.candidates) == 2
    assert spec.candidates[0].training_count == 2
    assert (
        deserialize_morphology_bundle_reranker_spec(
            serialize_morphology_bundle_reranker_spec(spec)
        )
        == spec
    )


def test_bundle_reranker_supports_strict_export() -> None:
    reranker = MorphologyBundleReranker(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=_schema(),
        spec=_spec(),
        dropout_probability=0.0,
    )
    inputs = {
        "hidden_states": torch.randn((1, 2, 4)),
        "upos_logits": torch.randn((1, 2, 2)),
        "morphology_logits": (
            torch.randn((1, 2, 2)),
            torch.randn((1, 2, 3)),
        ),
    }

    eager = reranker(**inputs)
    exported = torch.export.export(
        reranker,
        (),
        kwargs=inputs,
        strict=True,
    ).module()(**inputs)

    for eager_logits, exported_logits in zip(eager, exported, strict=True):
        torch.testing.assert_close(exported_logits, eager_logits)
