from typing import Any

from torch import nn
from torch.optim import AdamW

from prism.training.config import SupervisedTrainingConfig


def _split_weight_decay_parameters(
    module: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []

    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    return decay_parameters, no_decay_parameters


def build_supervised_adamw_optimizer(
    *,
    backbone: nn.Module,
    task_heads: nn.Module,
    config: SupervisedTrainingConfig,
) -> AdamW:
    backbone_decay, backbone_no_decay = _split_weight_decay_parameters(backbone)
    task_heads_decay, task_heads_no_decay = _split_weight_decay_parameters(task_heads)

    parameter_groups: list[dict[str, Any]] = [
        {
            "name": "backbone_decay",
            "params": backbone_decay,
            "lr": config.backbone_learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "backbone_no_decay",
            "params": backbone_no_decay,
            "lr": config.backbone_learning_rate,
            "weight_decay": 0.0,
        },
        {
            "name": "task_heads_decay",
            "params": task_heads_decay,
            "lr": config.task_head_learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "task_heads_no_decay",
            "params": task_heads_no_decay,
            "lr": config.task_head_learning_rate,
            "weight_decay": 0.0,
        },
    ]

    if any(not group["params"] for group in parameter_groups):
        raise ValueError(
            "Backbone and task heads must provide decay and no-decay parameters."
        )

    return AdamW(parameter_groups)
