import torch

from prism.data import TokenTaskTargetBatch
from prism.evaluation.metrics import count_token_task_predictions
from prism.modeling.outputs import TokenTaskPredictionBatch


def test_count_token_task_predictions_ignores_padding() -> None:
    token_mask = torch.tensor([[True, True, False]])

    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor([[1, 0, 0]]),
        morphology_targets=(
            torch.tensor(
                [
                    [
                        [False, True],
                        [True, False],
                        [True, False],
                    ]
                ]
            ),
        ),
        lemma_rule_ids=torch.tensor([[0, 1, 0]]),
        lemma_rule_mask=torch.tensor([[True, True, False]]),
        token_mask=token_mask,
    )
    predictions = TokenTaskPredictionBatch(
        upos_ids=torch.tensor([[1, 1, 1]]),
        morphology_predictions=(
            torch.tensor(
                [
                    [
                        [False, True],
                        [False, True],
                        [False, True],
                    ]
                ]
            ),
        ),
        lemma_rule_ids=torch.tensor([[0, 0, 1]]),
        token_mask=token_mask,
    )

    counts = count_token_task_predictions(
        predictions=predictions,
        targets=targets,
    )

    assert counts.token_count.item() == 2
    assert counts.upos_correct_count.item() == 1

    assert counts.morphology_correct_counts[0].item() == 1
    assert counts.morphology_annotated_counts[0].item() == 1
    assert counts.morphology_annotated_correct_counts[0].item() == 1

    assert counts.lemma_target_count.item() == 2
    assert counts.lemma_rule_correct_count.item() == 1

    torch.testing.assert_close(
        counts.morphology_true_positive_counts[0],
        torch.tensor([0, 1]),
    )
    torch.testing.assert_close(
        counts.morphology_false_positive_counts[0],
        torch.tensor([0, 1]),
    )
    torch.testing.assert_close(
        counts.morphology_false_negative_counts[0],
        torch.tensor([1, 0]),
    )
