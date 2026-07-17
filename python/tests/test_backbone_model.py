from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from prism.modeling import PretrainedBackboneSpec, load_backbone_model


class BufferTestModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer(
            "position_table",
            torch.tensor([1.0]),
            persistent=False,
        )


def test_load_backbone_model_uses_pinned_spec() -> None:
    spec = PretrainedBackboneSpec(
        model_id="organization/model",
        revision="0123456789abcdef0123456789abcdef01234567",
        trust_remote_code=False,
    )

    model = Mock()

    with patch(
        "prism.modeling.backbones.AutoModel.from_pretrained",
        return_value=model,
    ) as from_pretrained:
        loaded_model = load_backbone_model(spec)

    assert loaded_model is model
    from_pretrained.assert_called_once_with(
        "organization/model",
        revision="0123456789abcdef0123456789abcdef01234567",
        trust_remote_code=False,
    )


def test_load_backbone_model_reinitializes_non_persistent_buffers() -> None:
    spec = PretrainedBackboneSpec(
        model_id="organization/custom-model",
        revision="0123456789abcdef0123456789abcdef01234567",
        trust_remote_code=True,
        reinitialize_non_persistent_buffers=True,
    )
    loaded_model = BufferTestModel(PretrainedConfig())

    with torch.no_grad():
        loaded_model.get_parameter("weight").fill_(7.0)
        loaded_model.get_buffer("position_table").fill_(float("nan"))

    loaded_model.eval()

    with patch(
        "prism.modeling.backbones.AutoModel.from_pretrained",
        return_value=loaded_model,
    ):
        reinitialized_model = load_backbone_model(spec)

    assert reinitialized_model is not loaded_model
    assert torch.equal(
        reinitialized_model.get_parameter("weight"),
        torch.tensor([7.0]),
    )
    assert torch.equal(
        reinitialized_model.get_buffer("position_table"),
        torch.tensor([1.0]),
    )
    assert not reinitialized_model.training
