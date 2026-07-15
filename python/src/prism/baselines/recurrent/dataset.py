import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from prism.conllu import Token
from prism.baselines.recurrent.vocabulary import (
    encode_sentence,
    encode_sentence_characters,
    encode_sentence_feature,
)

class PosDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
            self,
            sentences: list[list[Token]],
            word_vocabulary: dict[str, int],
            tag_vocabulary: dict[str, int],
    ) -> None:
        self.sentences = sentences
        self.word_vocabulary = word_vocabulary
        self.tag_vocabulary = tag_vocabulary

    def __len__(self) -> int:
        return len(self.sentences)
        
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        word_ids, tag_ids = encode_sentence(
            self.sentences[index],
            self.word_vocabulary,
            self.tag_vocabulary,
        )

        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
        )

def collate_sentences(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    word_sequences, tag_sequences = zip(*batch)

    padded_words = pad_sequence(
        word_sequences,
        batch_first=True,
        padding_value=0,
    )

    padded_tags = pad_sequence(
        tag_sequences,
        batch_first=True,
        padding_value=-100,
    )

    lengths = torch.tensor(
        [len(sequence) for sequence in word_sequences],
        dtype=torch.long
    )

    return padded_words, padded_tags, lengths

class CharacterPosDataset(
    Dataset[tuple[Tensor, Tensor, list[Tensor]]]
):
    def __init__(
        self,
        sentences: list[list[Token]],
        word_vocabulary: dict[str, int],
        tag_vocabulary: dict[str, int],
        character_vocabulary: dict[str, int],
    ) -> None:
        self.sentences = sentences
        self.word_vocabulary = word_vocabulary
        self.tag_vocabulary = tag_vocabulary
        self.character_vocabulary = character_vocabulary

    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        sentence = self.sentences[index]

        word_ids, tag_ids = encode_sentence(
            sentence,
            self.word_vocabulary,
            self.tag_vocabulary,
        )
        character_ids = encode_sentence_characters(
            sentence,
            self.character_vocabulary,
        )

        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            [
                torch.tensor(ids, dtype=torch.long)
                for ids in character_ids
            ],
        )
    
def collate_character_sentences(
    batch: list[tuple[Tensor, Tensor, list[Tensor]]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    word_sequences, tag_sequences, character_sequences = zip(*batch)

    padded_words = pad_sequence(
        word_sequences,
        batch_first=True,
        padding_value=0,
    )
    padded_tags = pad_sequence(
        tag_sequences,
        batch_first=True,
        padding_value=-100,
    )

    sentence_lengths = torch.tensor(
        [len(sequence) for sequence in word_sequences],
        dtype=torch.long,
    )

    maximum_word_length = max(
        len(characters)
        for sentence in character_sequences
        for characters in sentence
    )

    padded_characters = torch.zeros(
        (
            len(batch),
            padded_words.size(1),
            maximum_word_length,
        ),
        dtype=torch.long,
    )
    character_lengths = torch.zeros(
        (
            len(batch),
            padded_words.size(1)
        ),
        dtype=torch.long,
    )

    for sentence_index, sentence in enumerate(character_sequences):
        for token_index, characters in enumerate(sentence):
            length = len(characters)
            padded_characters[
                sentence_index,
                token_index,
                :length,
            ] = characters
            character_lengths[sentence_index, token_index] = length

    return (
        padded_words,
        padded_characters,
        padded_tags,
        sentence_lengths,
        character_lengths
    )

class CharacterFeatureDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, list[Tensor]]]
):
    def __init__(
        self,
        sentences: list[list[Token]],
        word_vocabulary: dict[str, int],
        tag_vocabulary: dict[str, int],
        character_vocabulary: dict[str, int],
        feature_name: str,
        feature_vocabulary: dict[str, int],
    ) -> None:
        self.sentences = sentences
        self.word_vocabulary = word_vocabulary
        self.tag_vocabulary = tag_vocabulary
        self.character_vocabulary = character_vocabulary
        self.feature_name = feature_name
        self.feature_vocabulary = feature_vocabulary

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        sentence = self.sentences[index]

        word_ids, tag_ids = encode_sentence(
            sentence,
            self.word_vocabulary,
            self.tag_vocabulary,
        )
        feature_ids = encode_sentence_feature(
            sentence,
            self.feature_name,
            self.feature_vocabulary,
        )
        character_ids = encode_sentence_characters(
            sentence,
            self.character_vocabulary,
        )

        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            torch.tensor(feature_ids, dtype=torch.long),
            [
                torch.tensor(ids, dtype=torch.long)
                for ids in character_ids
            ]
        )

def collate_character_feature_sentences(
    batch: list[
        tuple[Tensor, Tensor, Tensor, list[Tensor]]
    ],
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    (
        word_sequences,
        tag_sequences,
        feature_sequences,
        character_sequences,
    ) = zip(*batch)

    (
        padded_words,
        padded_characters,
        padded_tags,
        sentence_lengths,
        character_lengths,
    ) = collate_character_sentences([
        (words, tags, characters)
        for words, tags, characters in zip(
            word_sequences,
            tag_sequences,
            character_sequences,
        )
    ])

    padded_features = pad_sequence(
        feature_sequences,
        batch_first=True,
        padding_value=-100,
    )

    return (
        padded_words,
        padded_characters,
        padded_tags,
        padded_features,
        sentence_lengths,
        character_lengths
    )
