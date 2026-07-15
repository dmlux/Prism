from collections.abc import Iterable

import torch
from torch import Tensor, nn

from vexo.model import (
    BiLSTMPosTagger,
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