from types import SimpleNamespace
from unittest.mock import Mock

import torch

from prism.modeling import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
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
        subword_end_indices=torch.tensor(
            [[2]],
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


def test_contextualize_subwords_uses_learned_layer_aggregation() -> None:
    batch = TokenizedBatch(
        input_ids=torch.tensor([[1, 10]], dtype=torch.long),
        attention_mask=torch.tensor([[True, True]], dtype=torch.bool),
        first_subword_indices=torch.tensor([[1]], dtype=torch.long),
        subword_end_indices=torch.tensor([[2]], dtype=torch.long),
        token_mask=torch.tensor([[True]], dtype=torch.bool),
    )
    hidden_states = tuple(
        torch.full(
            (1, 2, 3),
            fill_value=float(value),
        )
        for value in range(5)
    )
    model = Mock(
        return_value=SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
        )
    )
    aggregation = BackboneLayerAggregation(
        strategy=BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR,
    )

    output = contextualize_subwords(
        model=model,
        batch=batch,
        layer_aggregation=aggregation,
    )

    torch.testing.assert_close(
        output.hidden_states,
        torch.full((1, 2, 3), 2.5),
    )
    assert model.call_args.kwargs["output_hidden_states"] is True
