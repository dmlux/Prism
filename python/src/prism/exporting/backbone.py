from torch import Tensor, nn
from transformers import PreTrainedModel

from prism.modeling import (
    BackboneLayerAggregation,
    BackboneLayerAggregationStrategy,
)


class BackboneExportAdapter(nn.Module):
    def __init__(
        self,
        *,
        backbone: PreTrainedModel,
        layer_aggregation: BackboneLayerAggregation | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_aggregation = layer_aggregation or BackboneLayerAggregation(
            strategy=BackboneLayerAggregationStrategy.LAST,
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.layer_aggregation.requires_hidden_states:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        return self.layer_aggregation(
            last_hidden_state=output.last_hidden_state,
            hidden_states=getattr(output, "hidden_states", None),
        )
