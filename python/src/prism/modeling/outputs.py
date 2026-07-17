from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextualizedSubwordBatch:
    hidden_states: Tensor

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 3:
            raise ValueError("Contextualized hidden states must have three dimensions.")
        if not self.hidden_states.is_floating_point():
            raise ValueError(
                "Contextualized hidden states must use a floating-point dtype."
            )
        if not torch.isfinite(self.hidden_states).all().item():
            raise ValueError(
                "Contextualized hidden states must contain only finite values."
            )
        if any(dimension <= 0 for dimension in self.hidden_states.shape):
            raise ValueError("Contextualized hidden-state dimensions must be positive.")

    @property
    def batch_size(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def max_subword_count(self) -> int:
        return self.hidden_states.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.hidden_states.shape[2]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextualizedTokenBatch:
    hidden_states: Tensor
    token_mask: Tensor

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 3:
            raise ValueError("Contextualized token states must have three dimensions.")
        if not self.hidden_states.is_floating_point():
            raise ValueError(
                "Contextualized token states must use a floating-point dtype."
            )
        if self.token_mask.shape != self.hidden_states.shape[:2]:
            raise ValueError("Token mask shape must match contextualized token states.")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("Token mask use torch.bool.")

    @property
    def batch_size(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def max_token_count(self) -> int:
        return self.hidden_states.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.hidden_states.shape[2]
