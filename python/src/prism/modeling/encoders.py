from torch import Tensor, nn

from prism.modeling.batches import TokenizedBatch
from prism.modeling.layer_aggregation import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)
from prism.modeling.outputs import ContextualizedSubwordBatch


def contextualize_subwords(
    *,
    model: nn.Module,
    batch: TokenizedBatch,
    layer_aggregation: BackboneLayerAggregation | None = None,
) -> ContextualizedSubwordBatch:
    if layer_aggregation is None:
        layer_aggregation = BackboneLayerAggregation(
            strategy=BackboneLayerAggregationStrategy.LAST
        )

    if layer_aggregation.requires_hidden_states:
        raw_output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    else:
        raw_output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )

    last_hidden_state = getattr(
        raw_output,
        "last_hidden_state",
        None,
    )
    if not isinstance(last_hidden_state, Tensor):
        raise TypeError("Backbone output must provide tensor last_hidden_state.")

    raw_hidden_states = getattr(
        raw_output,
        "hidden_states",
        None,
    )
    if raw_hidden_states is not None and (
        not isinstance(raw_hidden_states, (list, tuple))
        or any(
            not isinstance(hidden_state, Tensor) for hidden_state in raw_hidden_states
        )
    ):
        raise TypeError("Backbone hidden states must be a sequence of tensors.")

    hidden_states = layer_aggregation(
        last_hidden_state=last_hidden_state,
        hidden_states=raw_hidden_states,
    )

    output = ContextualizedSubwordBatch(
        hidden_states=hidden_states,
    )

    if output.batch_size != batch.batch_size:
        raise ValueError("Backbone output batch size must match the tokenized batch.")
    if output.max_subword_count != batch.max_subword_count:
        raise ValueError(
            "Backbone output subword count must match the tokenized batch."
        )

    return output
