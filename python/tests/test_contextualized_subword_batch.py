import torch
import pytest

from prism.modeling import ContextualizedSubwordBatch


def test_contextualized_subword_batch_exposes_dimensions() -> None:
    batch = ContextualizedSubwordBatch(
        hidden_states=torch.zeros(
            (2, 5, 192),
            dtype=torch.float32,
        )
    )

    assert batch.batch_size == 2
    assert batch.max_subword_count == 5
    assert batch.hidden_size == 192


def test_contextualized_subword_batch_rejects_non_finite_values() -> None:
    hidden_states = torch.zeros((1, 2, 3), dtype=torch.float32)
    hidden_states[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        ContextualizedSubwordBatch(hidden_states=hidden_states)
