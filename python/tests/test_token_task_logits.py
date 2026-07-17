import torch

from prism.modeling import TokenTaskLogits


def test_token_task_logits_expose_shared_token_dimension() -> None:
    logits = TokenTaskLogits(
        upos_logits=torch.zeros((2, 4, 17)),
        morphology_logits=(
            torch.zeros((2, 4, 3)),
            torch.zeros((2, 4, 5)),
        ),
        lemma_rule_logits=torch.zeros((2, 4, 7)),
    )

    assert logits.batch_size == 2
    assert logits.max_token_count == 4
    assert logits.morphology_feature_count == 2
