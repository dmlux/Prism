from collections.abc import Sequence
from enum import StrEnum

import torch
from torch import Tensor, nn


class BackboneLayerAggregationStrategy(StrEnum):
    LAST = "last"
    LEARNED_LAST_FOUR = "learned-last-four"


class BackboneLayerAggregation(nn.Module):
    def __init__(
        self,
        *,
        strategy: BackboneLayerAggregationStrategy,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        if strategy is BackboneLayerAggregationStrategy.LAST:
            self.register_parameter("mixing_logits", None)
            self.register_parameter("scale", None)
        elif strategy is BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR:
            self.mixing_logits = nn.Parameter(torch.zeros(4))
            self.scale = nn.Parameter(torch.ones(()))
        else:
            raise ValueError(
                f"Unsupported backbone layer aggregation strategy: {strategy!r}"
            )

    @property
    def requires_hidden_states(self) -> bool:
        return self.strategy is BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR

    def forward(
        self,
        *,
        last_hidden_state: Tensor,
        hidden_states: Sequence[Tensor] | None = None,
    ) -> Tensor:
        if self.strategy is BackboneLayerAggregationStrategy.LAST:
            return last_hidden_state

        if hidden_states is None or len(hidden_states) < 4:
            raise ValueError(
                "Learned last-four aggregation requires at least four backbone "
                "hidden states."
            )
        if self.mixing_logits is None or self.scale is None:
            raise RuntimeError("Learned layer aggregation parameters are missing.")

        selected_hidden_states = hidden_states[-4:]
        if any(
            hidden_state.shape != last_hidden_state.shape
            for hidden_state in selected_hidden_states
        ):
            raise ValueError(
                "Aggregated backbone hidden states must match the final hidden-state "
                "shape."
            )

        mixing_weights = torch.softmax(
            self.mixing_logits,
            dim=0,
        )
        stacked_hidden_states = torch.stack(
            tuple(selected_hidden_states),
            dim=0,
        )
        mixed_hidden_states = torch.sum(
            stacked_hidden_states * mixing_weights[:, None, None, None],
            dim=0,
        )

        return self.scale * mixed_hidden_states
