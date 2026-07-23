import pytest
import torch

from prism.modeling import (
    MorphologyAgreementRefiner,
    MorphologyAgreementRefinerSpec,
)
from prism.schema import MorphologyFeatureSchema, MorphologySchema


MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Gender",
            values=("Fem", "Masc", "Neut"),
            allows_multiple_values=False,
        ),
        MorphologyFeatureSchema(
            name="Number",
            values=("Plur", "Sing"),
            allows_multiple_values=False,
        ),
        MorphologyFeatureSchema(
            name="PronType",
            values=("Art", "Prs"),
            allows_multiple_values=True,
        ),
    ),
)


def test_agreement_refiner_starts_as_neutral_gated_residual() -> None:
    refiner = MorphologyAgreementRefiner(
        hidden_size=8,
        upos_label_count=3,
        morphology_schema=MORPHOLOGY_SCHEMA,
        spec=MorphologyAgreementRefinerSpec(
            window_radius=3,
            bottleneck_size=4,
            target_feature_names=("Gender", "Number"),
        ),
        dropout_probability=0.0,
    )
    hidden_states = torch.randn((2, 5, 8))
    token_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    upos_logits = torch.randn((2, 5, 3))
    morphology_logits = (
        torch.randn((2, 5, 4)),
        torch.randn((2, 5, 3)),
        torch.randn((2, 5, 2)),
    )

    refined = refiner(
        hidden_states=hidden_states,
        token_mask=token_mask,
        upos_logits=upos_logits,
        morphology_logits=morphology_logits,
    )

    assert refiner.target_feature_indices == (0, 1)
    assert len(refined) == len(morphology_logits)
    for original, output in zip(morphology_logits, refined, strict=True):
        torch.testing.assert_close(output, original)

    refined[0].sum().backward()
    assert refiner.correction_heads[0].weight.grad is not None


def test_agreement_refiner_preserves_padding_and_non_target_feature() -> None:
    refiner = MorphologyAgreementRefiner(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=MORPHOLOGY_SCHEMA,
        spec=MorphologyAgreementRefinerSpec(
            window_radius=3,
            bottleneck_size=4,
            target_feature_names=("Gender",),
        ),
        dropout_probability=0.0,
    )
    with torch.no_grad():
        refiner.correction_heads[0].bias.fill_(1.0)

    morphology_logits = (
        torch.zeros((1, 3, 4)),
        torch.zeros((1, 3, 3)),
        torch.zeros((1, 3, 2)),
    )
    refined = refiner(
        hidden_states=torch.randn((1, 3, 4)),
        token_mask=torch.tensor([[True, True, False]]),
        upos_logits=torch.randn((1, 3, 2)),
        morphology_logits=morphology_logits,
    )

    assert torch.count_nonzero(refined[0][0, :2]).item() > 0
    torch.testing.assert_close(refined[0][0, 2], morphology_logits[0][0, 2])
    torch.testing.assert_close(refined[1], morphology_logits[1])
    torch.testing.assert_close(refined[2], morphology_logits[2])

    refiner.set_enabled(False)
    disabled = refiner(
        hidden_states=torch.randn((1, 3, 4)),
        token_mask=torch.tensor([[True, True, False]]),
        upos_logits=torch.randn((1, 3, 2)),
        morphology_logits=morphology_logits,
    )
    assert disabled is morphology_logits


def test_agreement_refiner_does_not_apply_a_bias_without_a_neighbor() -> None:
    refiner = MorphologyAgreementRefiner(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=MORPHOLOGY_SCHEMA,
        spec=MorphologyAgreementRefinerSpec(
            window_radius=3,
            bottleneck_size=4,
            target_feature_names=("Gender",),
        ),
        dropout_probability=0.0,
    )
    with torch.no_grad():
        refiner.correction_heads[0].bias.fill_(1.0)

    morphology_logits = (
        torch.zeros((1, 1, 4)),
        torch.zeros((1, 1, 3)),
        torch.zeros((1, 1, 2)),
    )
    refined = refiner(
        hidden_states=torch.randn((1, 1, 4)),
        token_mask=torch.tensor([[True]]),
        upos_logits=torch.randn((1, 1, 2)),
        morphology_logits=morphology_logits,
    )

    for original, output in zip(morphology_logits, refined, strict=True):
        torch.testing.assert_close(output, original)


def test_agreement_refiner_rejects_unknown_target_feature() -> None:
    with pytest.raises(ValueError, match="Unknown agreement target feature"):
        MorphologyAgreementRefiner(
            hidden_size=4,
            upos_label_count=2,
            morphology_schema=MORPHOLOGY_SCHEMA,
            spec=MorphologyAgreementRefinerSpec(
                window_radius=3,
                bottleneck_size=4,
                target_feature_names=("Unknown",),
            ),
            dropout_probability=0.0,
        )
