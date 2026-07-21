import torch

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenizedBatch
from prism.training import SupervisedTokenTaskBatch


def test_supervised_token_task_batch_connects_inputs_and_targets() -> None:
    model_inputs = TokenizedBatch(
        input_ids=torch.tensor(
            [[1, 10, 20, 2]],
            dtype=torch.long,
        ),
        attention_mask=torch.tensor(
            [[True, True, True, True]],
            dtype=torch.bool,
        ),
        first_subword_indices=torch.tensor(
            [[1, 2]],
            dtype=torch.long,
        ),
        subword_end_indices=torch.tensor(
            [[2, 3]],
            dtype=torch.long,
        ),
        token_mask=torch.tensor(
            [[True, True]],
            dtype=torch.bool,
        ),
    )
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor(
            [[0, 1]],
            dtype=torch.long,
        ),
        morphology_targets=(
            torch.tensor(
                [[[True, False], [False, True]]],
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor(
            [[0, 1]],
            dtype=torch.long,
        ),
        lemma_rule_mask=torch.tensor(
            [[True, True]],
            dtype=torch.bool,
        ),
        token_mask=torch.tensor(
            [[True, True]],
            dtype=torch.bool,
        ),
    )

    batch = SupervisedTokenTaskBatch(
        model_inputs=model_inputs,
        targets=targets,
    )

    assert batch.batch_size == 1
    assert batch.max_token_count == 2

    device = torch.device("cpu")
    moved_batch = batch.to(device)

    assert moved_batch is not batch
    assert moved_batch.model_inputs.input_ids.device == device
    assert moved_batch.model_inputs.attention_mask.device == device
    assert moved_batch.model_inputs.first_subword_indices.device == device
    assert moved_batch.model_inputs.subword_end_indices.device == device
    assert moved_batch.model_inputs.token_mask.device == device
    assert moved_batch.targets.upos_ids.device == device
    assert moved_batch.targets.morphology_targets[0].device == device
    assert moved_batch.targets.lemma_rule_ids.device == device
    assert moved_batch.targets.lemma_rule_mask.device == device
    assert moved_batch.targets.token_mask.device == device
