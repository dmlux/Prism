from collections.abc import Iterable

import torch
from torch import Tensor, nn

from prism.model import (
    BiLSTMPosTagger,
    CharacterBiLSTMMultiTaskTagger,
    CharacterBiLSTMPosTagger,
)

Batch = tuple[Tensor, Tensor, Tensor]

CharacterBatch = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]

MultiTaskBatch = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]

def train_epoch(
    model: BiLSTMPosTagger,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    batch_count = 0

    for word_ids, tag_ids, lengths in batches:
        word_ids = word_ids.to(device)
        tag_ids = tag_ids.to(device)

        optimizer.zero_grad()

        outputs = model(word_ids, lengths)
        loss = loss_function(
            outputs.reshape(-1, outputs.size(-1)),
            tag_ids.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count

def train_character_epoch(
    model: CharacterBiLSTMPosTagger,
    batches: Iterable[CharacterBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    batch_count = 0

    for (
        word_ids,
        character_ids,
        tag_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        word_ids = word_ids.to(device)
        character_ids = character_ids.to(device)
        tag_ids = tag_ids.to(device)

        optimizer.zero_grad()

        outputs = model(
            word_ids,
            character_ids,
            sentence_lengths,
            character_lengths,
        )
        loss = loss_function(
            outputs.reshape(-1, outputs.size(-1)),
            tag_ids.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count

def train_multitask_epoch(
    model: CharacterBiLSTMMultiTaskTagger,
    batches: Iterable[MultiTaskBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    feature_loss_weight: float = 1.0,
) -> tuple[float, float, float]:
    model.train()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_pos_loss = 0.0
    total_feature_loss = 0.0
    batch_count = 0

    for (
        word_ids,
        character_ids,
        tag_ids,
        feature_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        word_ids = word_ids.to(device)
        character_ids = character_ids.to(device)
        tag_ids = tag_ids.to(device)
        feature_ids = feature_ids.to(device)

        optimizer.zero_grad()

        pos_outputs, feature_outputs = model(
            word_ids,
            character_ids,
            sentence_lengths,
            character_lengths,
        )

        pos_loss = loss_function(
            pos_outputs.reshape(
                -1,
                pos_outputs.size(-1),
            ),
            tag_ids.reshape(-1),
        )
        feature_loss = loss_function(
            feature_outputs.reshape(
                -1,
                feature_outputs.size(-1),
            ),
            feature_ids.reshape(-1),
        )

        loss = (
            pos_loss
            + feature_loss_weight * feature_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pos_loss += pos_loss.item()
        total_feature_loss += feature_loss.item()
        batch_count += 1

    return (
        total_loss / batch_count,
        total_pos_loss / batch_count,
        total_feature_loss / batch_count,
    )

@torch.no_grad()
def evaluate(
    model: BiLSTMPosTagger,
    batches: Iterable[Batch],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    token_count = 0

    for word_ids, tag_ids, lengths in batches:
        word_ids = word_ids.to(device)
        tag_ids = tag_ids.to(device)

        outputs = model(word_ids, lengths)
        loss = loss_function(
            outputs.reshape(-1, outputs.size(-1)),
            tag_ids.reshape(-1),
        )

        valid = tag_ids != -100
        predictions = outputs.argmax(dim=-1)

        valid_count = valid.sum().item()
        total_loss += loss.item() * valid_count
        correct += ((predictions == tag_ids) & valid).sum().item()
        token_count += valid_count
    
    return total_loss / token_count, correct / token_count

@torch.no_grad()
def evaluate_character(
    model: CharacterBiLSTMPosTagger,
    batches: Iterable[CharacterBatch],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    token_count = 0

    for (
        word_ids,
        character_ids,
        tag_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        word_ids = word_ids.to(device)
        character_ids = character_ids.to(device)
        tag_ids = tag_ids.to(device)

        outputs = model(
            word_ids,
            character_ids,
            sentence_lengths,
            character_lengths
        )
        loss = loss_function(
            outputs.reshape(-1, outputs.size(-1)),
            tag_ids.reshape(-1),
        )

        valid = tag_ids != -100
        predictions = outputs.argmax(dim=-1)

        valid_count = valid.sum().item()
        total_loss += loss.item() * valid_count
        correct += (
            (predictions == tag_ids) & valid
        ).sum().item()
        token_count += valid_count

    return total_loss / token_count, correct / token_count

@torch.no_grad()
def evaluate_character_knownness(
    model: CharacterBiLSTMPosTagger,
    batches: Iterable[CharacterBatch],
    device: torch.device,
    unknown_word_id: int,
) -> tuple[int, int, int, int]:
    model.eval()

    known_correct = 0
    known_count = 0
    unknown_correct = 0
    unknown_count = 0

    for (
        word_ids,
        character_ids,
        tag_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        word_ids = word_ids.to(device)
        character_ids = character_ids.to(device)
        tag_ids = tag_ids.to(device)

        outputs = model(
            word_ids,
            character_ids,
            sentence_lengths,
            character_lengths,
        )
        predictions = outputs.argmax(dim=-1)

        valid = tag_ids != -100
        unknown = valid & (word_ids == unknown_word_id)
        known = valid & ~unknown

        known_correct += (
            (predictions == tag_ids) & known
        ).sum().item()
        known_count += known.sum().item()

        unknown_correct += (
            (predictions == tag_ids) & unknown
        ).sum().item()
        unknown_count += unknown.sum().item()

    return (
        known_correct,
        known_count,
        unknown_correct,
        unknown_count
    )

@torch.no_grad()
def evaluate_knownness(
    model: BiLSTMPosTagger,
    batches: Iterable[Batch],
    device: torch.device,
    unknown_word_id: int,
) -> tuple[int, int, int, int]:
    model.eval()

    known_correct = 0
    known_count = 0
    unknown_correct = 0
    unknown_count = 0

    for word_ids, tag_ids, lengths in batches:
        word_ids = word_ids.to(device)
        tag_ids = tag_ids.to(device)

        outputs = model(word_ids, lengths)
        predictions = outputs.argmax(dim=-1)

        valid = tag_ids != -100
        unknown = valid & (word_ids == unknown_word_id)
        known = valid & ~unknown

        known_correct += (
            (predictions == tag_ids) & known
        ).sum().item()
        known_count += known.sum().item()

        unknown_correct += (
            (predictions == tag_ids) & unknown
        ).sum().item()
        unknown_count += unknown.sum().item()

    return (
        known_correct,
        known_count,
        unknown_correct,
        unknown_count,
    )

@torch.no_grad()
def character_confusion_matrix(
    model: CharacterBiLSTMPosTagger,
    batches: Iterable[CharacterBatch],
    device: torch.device,
    tag_count: int,
) -> Tensor:
    model.eval()

    matrix = torch.zeros(
        (tag_count, tag_count),
        dtype=torch.long,
    )

    for (
        word_ids,
        character_ids,
        tag_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        outputs = model(
            word_ids.to(device),
            character_ids.to(device),
            sentence_lengths,
            character_lengths,
        )
        predictions = outputs.argmax(dim=-1).cpu()
        valid = tag_ids != -100

        actual_tags = tag_ids[valid]
        predicted_tags = predictions[valid]

        for actual, predicted in zip(
            actual_tags.tolist(),
            predicted_tags.tolist(),
        ):
            matrix[actual, predicted] += 1

    return matrix


@torch.no_grad()
def evaluate_multitask(
    model: CharacterBiLSTMMultiTaskTagger,
    batches: Iterable[MultiTaskBatch],
    device: torch.device,
    no_feature_id: int,
) -> tuple[float, float, float, float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    total_pos_loss = 0.0
    total_feature_loss = 0.0

    pos_correct = 0
    pos_count = 0

    feature_correct = 0
    feature_count = 0

    annotated_correct = 0
    annotated_count = 0

    for (
        word_ids,
        character_ids,
        tag_ids,
        feature_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        word_ids = word_ids.to(device)
        character_ids = character_ids.to(device)
        tag_ids = tag_ids.to(device)
        feature_ids = feature_ids.to(device)

        pos_outputs, feature_outputs = model(
            word_ids,
            character_ids,
            sentence_lengths,
            character_lengths
        )

        pos_loss = loss_function(
            pos_outputs.reshape(
                -1,
                pos_outputs.size(-1),
            ),
            tag_ids.reshape(-1),
        )
        feature_loss = loss_function(
            feature_outputs.reshape(
                -1,
                feature_outputs.size(-1),
            ),
            feature_ids.reshape(-1),
        )

        pos_valid = tag_ids != -100
        feature_valid = feature_ids != -100
        annotated = (
            feature_valid
            & (feature_ids != no_feature_id)
        )

        pos_predictions = pos_outputs.argmax(dim=-1)
        feature_predictions = feature_outputs.argmax(dim=-1)

        current_pos_count = pos_valid.sum().item()
        current_feature_count = feature_valid.sum().item()

        total_pos_loss += (
            pos_loss.item() * current_pos_count
        )
        total_feature_loss += (
            feature_loss.item() * current_feature_count
        )

        pos_correct += (
            (pos_predictions == tag_ids) & pos_valid
        ).sum().item()
        pos_count += current_pos_count

        feature_correct += (
            (feature_predictions == feature_ids)
            & feature_valid
        ).sum().item()
        feature_count += current_feature_count

        annotated_correct += (
            (feature_predictions == feature_ids)
            & annotated
        ).sum().item()
        annotated_count += annotated.sum().item()

    return (
        total_pos_loss / pos_count,
        pos_correct / pos_count,
        total_feature_loss / feature_count,
        feature_correct / feature_count,
        annotated_correct / annotated_count
    )

@torch.no_grad()
def multitask_feature_confusion_matrix(
    model: CharacterBiLSTMMultiTaskTagger,
    batches: Iterable[MultiTaskBatch],
    device: torch.device,
    feature_count: int,
) -> Tensor:
    model.eval()

    matrix = torch.zeros(
        (feature_count, feature_count),
        dtype=torch.long,
    )

    for (
        word_ids,
        character_ids,
        tag_ids,
        feature_ids,
        sentence_lengths,
        character_lengths,
    ) in batches:
        _, feature_outputs = model(
            word_ids.to(device),
            character_ids.to(device),
            sentence_lengths,
            character_lengths,
        )

        predictions = feature_outputs.argmax(
            dim=-1
        ).cpu()
        valid = feature_ids != -100

        actual_values = feature_ids[valid]
        predicted_values = predictions[valid]

        for actual, predicted in zip(
            actual_values.tolist(),
            predicted_values.tolist(),
        ):
            matrix[actual, predicted] += 1

    return matrix