import torch

from prism.modeling.decoding import decode_token_task_logits
from prism.modeling.outputs import TokenTaskLogits
from prism.schema import (
    MorphologyFeatureSchema,
    MorphologySchema,
)


def test_decode_token_task_logits_produces_valid_predictions() -> None:
    morphology_schema = MorphologySchema(
        version=1,
        features=(
            MorphologyFeatureSchema(
                name="Case",
                values=("Acc", "Nom"),
                allows_multiple_values=False,
            ),
            MorphologyFeatureSchema(
                name="PronType",
                values=("Art", "Dem"),
                allows_multiple_values=True,
            ),
        ),
    )
    logits = TokenTaskLogits(
        upos_logits=torch.tensor([[[0.0, 2.0], [3.0, 1.0]]]),
        morphology_logits=(
            torch.tensor([[[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]]]),
            torch.tensor(
                [
                    [
                        [0.0, 2.0, 1.0],
                        [-3.0, -0.1, -2.0],
                    ]
                ]
            ),
        ),
        lemma_rule_logits=torch.tensor([[[2.0, 0.0], [0.0, 4.0]]]),
    )
    token_mask = torch.tensor([[True, True]])

    predictions = decode_token_task_logits(
        logits=logits,
        token_mask=token_mask,
        morphology_schema=morphology_schema,
    )

    torch.testing.assert_close(
        predictions.upos_ids,
        torch.tensor([[1, 0]]),
    )
    torch.testing.assert_close(
        predictions.morphology_predictions[0],
        torch.tensor([[[False, True, False], [True, False, False]]]),
    )
    torch.testing.assert_close(
        predictions.morphology_predictions[1],
        torch.tensor([[[False, True, True], [True, False, False]]]),
    )
    torch.testing.assert_close(
        predictions.lemma_rule_ids,
        torch.tensor([[0, 1]]),
    )
    torch.testing.assert_close(
        predictions.token_mask,
        token_mask,
    )
