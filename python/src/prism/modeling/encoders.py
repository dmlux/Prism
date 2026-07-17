from torch import Tensor, nn

from prism.modeling.batches import TokenizedBatch
from prism.modeling.outputs import ContextualizedSubwordBatch


def contextualize_subwords(
    *,
    model: nn.Module,
    batch: TokenizedBatch,
) -> ContextualizedSubwordBatch:
    raw_output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        return_dict=True,
    )

    hidden_states = getattr(
        raw_output,
        "last_hidden_state",
        None,
    )
    if not isinstance(hidden_states, Tensor):
        raise TypeError("Backbone output must provide tensor last_hidden_state.")

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
