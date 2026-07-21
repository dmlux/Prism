import torch
from torch import Tensor, nn

from prism.modeling.character_batches import CharacterTokenBatch
from prism.schema import CHARACTER_PADDING_ID


class CharacterCnnTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        hidden_size: int,
        embedding_size: int = 32,
    ) -> None:
        super().__init__()

        if vocabulary_size <= CHARACTER_PADDING_ID + 1:
            raise ValueError("Character vocabulary size must include literal values.")
        if hidden_size <= 1:
            raise ValueError("Character hidden size must be greater than one.")
        if embedding_size <= 0:
            raise ValueError("Character embedding size must be positive.")

        narrow_channel_count = hidden_size // 2
        wide_channel_count = hidden_size - narrow_channel_count
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_size,
            padding_idx=CHARACTER_PADDING_ID,
        )
        self.narrow_convolution = nn.Conv1d(
            embedding_size,
            narrow_channel_count,
            kernel_size=3,
            padding=1,
        )
        self.wide_convolution = nn.Conv1d(
            embedding_size,
            wide_channel_count,
            kernel_size=5,
            padding=2,
        )
        self.activation = nn.GELU()

    def forward(self, batch: CharacterTokenBatch) -> Tensor:
        batch_size, token_count, character_count = batch.character_ids.shape
        embedded = self.embedding(batch.character_ids).reshape(
            batch_size * token_count,
            character_count,
            -1,
        )
        convolution_input = embedded.transpose(1, 2)
        encoded = torch.cat(
            (
                self.activation(self.narrow_convolution(convolution_input)),
                self.activation(self.wide_convolution(convolution_input)),
            ),
            dim=1,
        ).transpose(1, 2)
        character_mask = batch.character_mask.reshape(
            batch_size * token_count,
            character_count,
            1,
        )
        masked = encoded.masked_fill(~character_mask, torch.finfo(encoded.dtype).min)
        pooled = masked.amax(dim=1).reshape(batch_size, token_count, -1)

        return pooled.masked_fill(~batch.token_mask.unsqueeze(-1), 0.0)


class CharacterResidualFusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("Fusion hidden size must be positive.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be greater than or equal to zero "
                "and less than one."
            )

        self.normalization = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)

    def forward(
        self,
        *,
        contextual_hidden_states: Tensor,
        character_hidden_states: Tensor,
    ) -> Tensor:
        if contextual_hidden_states.shape != character_hidden_states.shape:
            raise ValueError("Contextual and character hidden-state shapes must match.")

        normalized_characters = self.normalization(character_hidden_states)
        correction = self.projection(
            torch.cat(
                (contextual_hidden_states, normalized_characters),
                dim=-1,
            )
        )

        return contextual_hidden_states + self.dropout(self.activation(correction))
