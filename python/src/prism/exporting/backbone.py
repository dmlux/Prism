from torch import Tensor, nn
from transformers import PreTrainedModel


class BackboneExportAdapter(nn.Module):
    def __init__(self, backbone: PreTrainedModel) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state
