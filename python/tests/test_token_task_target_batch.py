import torch

from prism.data import TokenTaskTargetBatch


def test_token_task_target_batch_exposes_training_dimensions() -> None:
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor(
            [[0, 1, 0]],
            dtype=torch.long,
        ),
        morphology_targets=(
            torch.tensor(
                [[[False, True], [True, False], [False, False]]],
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor(
            [[1, 0, 0]],
            dtype=torch.long,
        ),
        lemma_rule_mask=torch.tensor(
            [[True, False, False]],
            dtype=torch.bool,
        ),
        token_mask=torch.tensor(
            [[True, True, False]],
            dtype=torch.bool,
        ),
    )

    assert targets.batch_size == 1
    assert targets.max_token_count == 3
    assert targets.morphology_feature_count == 1
