from types import SimpleNamespace
from unittest.mock import Mock

import torch

from prism.modeling import (
    TokenizedBatch,
    contextualize_subwords,
)


def test_contextualize_subwords_wraps_backbone_output() -> None:
    batch = TokenizedBatch(
        input_ids=torch.tensor(
            [[1, 10]],
            dtype=torch.long,
        ),
        attention_mask=torch.tensor(
            [[True, True]],
            dtype=torch.bool,
        ),
        first_subword_indices=torch.tensor(
            [[1]],
            dtype=torch.long,
        ),
        token_mask=torch.tensor(
            [[True]],
            dtype=torch.bool,
        ),
    )
    hidden_states = torch.zeros(
        (1, 2, 192),
        dtype=torch.float32,
    )
    model = Mock(
        return_value=SimpleNamespace(
            last_hidden_state=hidden_states,
        )
    )

    output = contextualize_subwords(
        model=model,
        batch=batch,
    )

    assert output.hidden_states is hidden_states

    model.assert_called_once()
    call = model.call_args
    assert call.kwargs["input_ids"] is batch.input_ids
    assert call.kwargs["attention_mask"] is batch.attention_mask
    assert call.kwargs["return_dict"] is True
