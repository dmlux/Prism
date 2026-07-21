import torch

from prism.modeling import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)


def test_last_layer_aggregation_preserves_final_hidden_state() -> None:
    aggregation = BackboneLayerAggregation(
        strategy=BackboneLayerAggregationStrategy.LAST,
    )
    last_hidden_state = torch.randn((2, 3, 4))

    result = aggregation(
        last_hidden_state=last_hidden_state,
    )

    assert result is last_hidden_state
    assert tuple(aggregation.parameters()) == ()
    assert not aggregation.requires_hidden_states


def test_learned_last_four_aggregation_starts_as_uniform_mix() -> None:
    aggregation = BackboneLayerAggregation(
        strategy=BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR,
    )
    hidden_states = tuple(
        torch.full(
            (1, 2, 3),
            fill_value=float(value),
        )
        for value in range(5)
    )

    result = aggregation(
        last_hidden_state=hidden_states[-1],
        hidden_states=hidden_states,
    )

    torch.testing.assert_close(
        result,
        torch.full((1, 2, 3), 2.5),
    )
    assert aggregation.requires_hidden_states
    assert sum(parameter.numel() for parameter in aggregation.parameters()) == 5

    result.sum().backward()

    assert aggregation.mixing_logits is not None
    assert aggregation.mixing_logits.grad is not None
    assert aggregation.scale is not None
    assert aggregation.scale.grad is not None
