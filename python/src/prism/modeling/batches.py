from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedBatch:
    input_ids: Tensor
    attention_mask: Tensor
    first_subword_indices: Tensor
    subword_end_indices: Tensor
    token_mask: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("Input IDs must have two dimensions.")
        if self.input_ids.dtype != torch.long:
            raise ValueError("Input IDs must use torch.long")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("Attention mask shape must match input IDs.")
        if self.attention_mask.dtype != torch.bool:
            raise ValueError("Attention mask must use torch.bool.")
        if self.first_subword_indices.ndim != 2:
            raise ValueError("First-subword indices must have two dimensions.")
        if self.first_subword_indices.dtype != torch.long:
            raise ValueError("First-subword indices must use torch.long.")
        if self.subword_end_indices.shape != self.first_subword_indices.shape:
            raise ValueError("Subword-end indices must match first-subword indices.")
        if self.subword_end_indices.dtype != torch.long:
            raise ValueError("Subword-end indices must use torch.long.")
        if self.token_mask.shape != self.first_subword_indices.shape:
            raise ValueError("Token mask shape must match first-subword indices.")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("Token mask must use torch.bool.")
        if self.input_ids.shape[0] != self.first_subword_indices.shape[0]:
            raise ValueError("Subword and token batch sizes must match.")

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def max_subword_count(self) -> int:
        return self.input_ids.shape[1]

    @property
    def max_token_count(self) -> int:
        return self.first_subword_indices.shape[1]

    def to(self, device: torch.device) -> "TokenizedBatch":
        return TokenizedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            first_subword_indices=self.first_subword_indices.to(device=device),
            subword_end_indices=self.subword_end_indices.to(device=device),
            token_mask=self.token_mask.to(device=device),
        )
