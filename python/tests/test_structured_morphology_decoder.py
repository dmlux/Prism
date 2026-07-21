import torch

from prism.modeling import StructuredMorphologyDecoder
from prism.schema import MorphologyFeatureSchema, MorphologySchema


def test_structured_morphology_decoder_refines_soft_decision_context() -> None:
    schema = MorphologySchema(
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
    decoder = StructuredMorphologyDecoder(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=schema,
        dropout_probability=0.0,
    )
    upos_logits = torch.tensor([[[10.0, -10.0]]])
    morphology_logits = (
        torch.zeros((1, 1, 2)),
        torch.zeros((1, 1, 3)),
    )

    initial_logits = decoder(
        upos_logits=upos_logits,
        morphology_logits=morphology_logits,
    )

    assert sum(parameter.numel() for parameter in decoder.parameters()) == 57
    for initial, base in zip(initial_logits, morphology_logits, strict=True):
        torch.testing.assert_close(initial, base)

    with torch.no_grad():
        decoder.context_projection.weight.zero_()
        decoder.context_projection.weight[:, 0] = 1.0
        decoder.context_projection.bias.zero_()
        for refinement_head in decoder.refinement_heads:
            refinement_head.weight.fill_(1.0)
            refinement_head.bias.zero_()

    noun_logits = decoder(
        upos_logits=upos_logits,
        morphology_logits=morphology_logits,
    )
    verb_logits = decoder(
        upos_logits=-upos_logits,
        morphology_logits=morphology_logits,
    )

    assert not torch.allclose(noun_logits[0], verb_logits[0])
    assert noun_logits[0].shape == (1, 1, 2)
    assert noun_logits[1].shape == (1, 1, 3)


def test_structured_morphology_decoder_supports_strict_export() -> None:
    schema = MorphologySchema(
        version=1,
        features=(
            MorphologyFeatureSchema(
                name="Number",
                values=("Plur", "Sing"),
                allows_multiple_values=False,
            ),
        ),
    )
    decoder = StructuredMorphologyDecoder(
        hidden_size=4,
        upos_label_count=2,
        morphology_schema=schema,
        dropout_probability=0.0,
    )
    inputs = {
        "upos_logits": torch.randn((1, 2, 2)),
        "morphology_logits": (torch.randn((1, 2, 3)),),
    }

    eager_outputs = decoder(**inputs)
    exported_outputs = torch.export.export(
        decoder,
        (),
        kwargs=inputs,
        strict=True,
    ).module()(**inputs)

    for eager, exported in zip(eager_outputs, exported_outputs, strict=True):
        torch.testing.assert_close(exported, eager)
