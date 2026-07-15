import torch

from pathlib import Path

from vexo.model import (
    BiLSTMPosTagger,
    CharacterBiLSTMPosTagger,
)
from vexo.vocabulary import (
    UNKNOWN_CHARACTER,
    UNKNOWN_TOKEN
)

def load_pos_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    BiLSTMPosTagger,
    dict[str, int],
    dict[str, int],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    word_vocabulary = checkpoint["word_vocabulary"]
    tag_vocabulary = checkpoint["tag_vocabulary"]

    model = BiLSTMPosTagger(
        vocabulary_size=len(word_vocabulary),
        tag_count=len(tag_vocabulary),
        embedding_size=checkpoint["embedding_size"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model, word_vocabulary, tag_vocabulary

def load_character_pos_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    CharacterBiLSTMPosTagger,
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    word_vocabulary = checkpoint["word_vocabulary"]
    character_vocabulary = checkpoint[
        "character_vocabulary"
    ]
    tag_vocabulary = checkpoint["tag_vocabulary"]

    model = CharacterBiLSTMPosTagger(
        vocabulary_size=len(word_vocabulary),
        character_count=len(character_vocabulary),
        tag_count=len(tag_vocabulary),
        word_embedding_size=checkpoint[
            "word_embedding_size"
        ],
        character_embedding_size=checkpoint[
            "character_embedding_size"
        ],
        character_hidden_size=checkpoint[
            "character_hidden_size"
        ],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return (
        model,
        word_vocabulary,
        character_vocabulary,
        tag_vocabulary,
    )

@torch.no_grad()
def predict_pos_tags(
    model: BiLSTMPosTagger,
    tokens: list[str],
    word_vocabulary: dict[str, int],
    tag_vocabulary: dict[str, int],
    device: torch.device,
) -> list[str]:
    model.eval()

    unknown_id = word_vocabulary[UNKNOWN_TOKEN]
    word_ids = [
        word_vocabulary.get(token, unknown_id)
        for token in tokens
    ]

    inputs = torch.tensor(
        [word_ids],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor([len(tokens)])

    outputs = model(inputs, lengths)
    predicted_ids = outputs.argmax(dim=-1)[0].tolist()

    id_to_tag = {
        identifier: tag
        for tag, identifier in tag_vocabulary.items()
    }

    return [
        id_to_tag[identifier]
        for identifier in predicted_ids
    ]

@torch.no_grad()
def predict_character_pos_tags(
    model: CharacterBiLSTMPosTagger,
    tokens: list[str],
    word_vocabulary: dict[str, int],
    character_vocabulary: dict[str, int],
    tag_vocabulary: dict[str, int],
    device: torch.device,
) -> list[str]:
    if not tokens:
        return []
    
    model.eval()

    unknown_word_id = word_vocabulary[UNKNOWN_TOKEN]
    unknown_character_id = character_vocabulary[
        UNKNOWN_CHARACTER
    ]

    word_ids = [
        word_vocabulary.get(token, unknown_word_id)
        for token in tokens
    ]

    maximum_word_length = max(
        len(token) for token in tokens
    )

    character_ids = torch.zeros(
        (1, len(tokens), maximum_word_length),
        dtype=torch.long,
    )
    character_lengths = torch.zeros(
        (1, len(tokens)),
        dtype=torch.long,
    )

    for token_index, token in enumerate(tokens):
        encoded = [
            character_vocabulary.get(
                character,
                unknown_character_id,
            )
            for character in token
        ]

        character_ids[
            0,
            token_index,
            :len(encoded),
        ] = torch.tensor(encoded)
        character_lengths[0, token_index] = len(encoded)

    inputs = torch.tensor(
        [word_ids],
        dtype=torch.long,
        device=device
    )
    character_ids = character_ids.to(device)
    sentence_lengths = torch.tensor([len(tokens)])

    outputs = model(
        inputs,
        character_ids,
        sentence_lengths,
        character_lengths
    )
    predicted_ids = outputs.argmax(dim=-1)[0].tolist()

    id_to_tag = {
        identifier: tag
        for tag, identifier in tag_vocabulary.items()
    }

    return [
        id_to_tag[identifier]
        for identifier in predicted_ids
    ]