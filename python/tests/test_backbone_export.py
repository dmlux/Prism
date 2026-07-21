from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from transformers import PreTrainedModel

from prism.exporting import BackboneExportAdapter
from prism.modeling import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)


def test_backbone_export_adapter_returns_hidden_state_tensor() -> None:
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[True, True, True]])
    hidden_states = torch.randn((1, 3, 192))

    backbone = Mock()
    backbone.return_value = SimpleNamespace(last_hidden_state=hidden_states)

    adapter = BackboneExportAdapter(
        backbone=cast(PreTrainedModel, backbone),
    )
    result = adapter(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert result is hidden_states
    backbone.assert_called_once_with(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )


def test_backbone_export_adapter_aggregates_hidden_states() -> None:
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[True, True, True]])
    hidden_states = tuple(torch.full((1, 3, 192), float(index)) for index in range(5))

    backbone = Mock()
    backbone.return_value = SimpleNamespace(
        last_hidden_state=hidden_states[-1],
        hidden_states=hidden_states,
    )
    layer_aggregation = BackboneLayerAggregation(
        strategy=BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR,
    )
    adapter = BackboneExportAdapter(
        backbone=cast(PreTrainedModel, backbone),
        layer_aggregation=layer_aggregation,
    )

    result = adapter(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    torch.testing.assert_close(
        result,
        torch.full((1, 3, 192), 2.5),
    )
    backbone.assert_called_once_with(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
