import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharacterEncoder(nn.Module):
    def __init__(
        self, character_count: int, embedding_size: int = 32, hidden_size: int = 32
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            character_count,
            embedding_size,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output_size = hidden_size * 2

    def forward(
        self,
        character_ids: Tensor,
        character_lengths: Tensor,
    ) -> Tensor:
        batch_size, sentence_length, word_length = character_ids.shape

        flattened_ids = character_ids.reshape(
            batch_size * sentence_length,
            word_length,
        )
        flattened_lengths = character_lengths.reshape(-1)

        embeddings = self.embedding(flattened_ids)

        packed = pack_padded_sequence(
            embeddings,
            flattened_lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden_states, _) = self.lstm(packed)

        representations = torch.cat(
            (hidden_states[0], hidden_states[1]),
            dim=-1,
        )

        valid_tokens = (flattened_lengths > 0).to(
            device=representations.device,
            dtype=representations.dtype,
        )
        representations = representations * valid_tokens.unsqueeze(-1)

        return representations.reshape(
            batch_size,
            sentence_length,
            self.output_size,
        )


class BiLSTMPosTagger(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        tag_count: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_size,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_size * 2, tag_count)

    def forward(
        self,
        word_ids: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        embeddings = self.embedding(word_ids)

        packed_embeddings = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_context, _ = self.lstm(packed_embeddings)

        contextualized, _ = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=word_ids.size(1),
        )

        return self.output(contextualized)


class CharacterBiLSTMPosTagger(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        character_count: int,
        tag_count: int,
        word_embedding_size: int = 64,
        character_embedding_size: int = 32,
        character_hidden_size: int = 32,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()

        self.word_embedding = nn.Embedding(
            vocabulary_size,
            word_embedding_size,
            padding_idx=0,
        )
        self.character_encoder = CharacterEncoder(
            character_count=character_count,
            embedding_size=character_embedding_size,
            hidden_size=character_hidden_size,
        )

        combine_size = word_embedding_size + self.character_encoder.output_size

        self.lstm = nn.LSTM(
            input_size=combine_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(
            hidden_size * 2,
            tag_count,
        )

    def encode(
        self,
        word_ids: Tensor,
        character_ids: Tensor,
        sentence_lengths: Tensor,
        character_lengths: Tensor,
    ) -> Tensor:
        word_representations = self.word_embedding(word_ids)
        character_representations = self.character_encoder(
            character_ids, character_lengths
        )

        combined = torch.cat(
            (
                word_representations,
                character_representations,
            ),
            dim=-1,
        )

        packed = pack_padded_sequence(
            combined,
            sentence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_context, _ = self.lstm(packed)

        contextualized, _ = pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=word_ids.size(1),
        )

        return contextualized

    def forward(
        self,
        word_ids: Tensor,
        character_ids: Tensor,
        sentence_lengths: Tensor,
        character_lengths: Tensor,
    ) -> Tensor:
        contextualized = self.encode(
            word_ids, character_ids, sentence_lengths, character_lengths
        )

        return self.output(contextualized)


class CharacterBiLSTMMultiTaskTagger(CharacterBiLSTMPosTagger):
    def __init__(
        self,
        vocabulary_size: int,
        character_count: int,
        tag_count: int,
        feature_count: int,
        word_embedding_size: int = 64,
        character_embedding_size: int = 32,
        character_hidden_size: int = 32,
        hidden_size: int = 128,
    ) -> None:
        super().__init__(
            vocabulary_size=vocabulary_size,
            character_count=character_count,
            tag_count=tag_count,
            word_embedding_size=word_embedding_size,
            character_embedding_size=character_embedding_size,
            character_hidden_size=character_hidden_size,
            hidden_size=hidden_size,
        )

        self.feature_output = nn.Linear(
            hidden_size * 2,
            feature_count,
        )

    def forward(
        self,
        word_ids: Tensor,
        character_ids: Tensor,
        sentence_lengths: Tensor,
        character_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        contextualized = self.encode(
            word_ids,
            character_ids,
            sentence_lengths,
            character_lengths,
        )

        pos_outputs = self.output(contextualized)
        feature_outputs = self.feature_output(contextualized)

        return pos_outputs, feature_outputs
