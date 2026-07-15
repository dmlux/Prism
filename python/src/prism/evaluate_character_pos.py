import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.conllu import read_sentences
from prism.dataset import (
    CharacterPosDataset,
    collate_character_sentences,
)
from prism.inference import load_character_pos_model
from prism.training import (
    character_confusion_matrix,
    evaluate_character,
    evaluate_character_knownness,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/norwegian-bokmaal/"
            "pos_character_bilstm.pt"
        ),
    )
    arguments = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    (
        model,
        words,
        characters,
        tags,
    ) = load_character_pos_model(
        arguments.checkpoint,
        device,
    )

    filename = (
        f"no_bokmaal-ud-{arguments.split}.conllu"
    )

    sentences = read_sentences(
        Path("data/raw/UD_Norwegian-Bokmaal") / filename
    )
    loader = DataLoader(
        CharacterPosDataset(
            sentences,
            words,
            tags,
            characters,
        ),
        batch_size=64,
        collate_fn=collate_character_sentences,
    )

    loss, accuracy = evaluate_character(
        model,
        loader,
        device,
    )

    (
        known_correct,
        known_count,
        unknown_correct,
        unknown_count,
    ) = evaluate_character_knownness(
        model,
        loader,
        device,
        words["<UNK>"],
    )

    matrix = character_confusion_matrix(
        model,
        loader,
        device,
        len(tags),
    )

    known_accuracy = known_correct / known_count
    unknown_accuracy = unknown_correct / unknown_count

    print("Split:", arguments.split)
    print("Sätze:", len(sentences))
    print("Loss:", f"{loss:.4f}")
    print("Genauigkeit:", f"{accuracy:.2%}")
    print(
        "Bekannte Tokens:",
        known_count,
        f"({known_accuracy:.2%} korrekt)",
    )
    print(
        "<UNK>-Tokens:",
        unknown_count,
        f"({unknown_accuracy:.2%} korrekt)",
    )
    print()
    print(
        f"{'POS':<8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )

    for tag, tag_id in sorted(
        tags.items(),
        key=lambda item: item[1],
    ):
        true_positive = matrix[tag_id, tag_id].item()
        predicted_count = matrix[:, tag_id].sum().item()
        actual_count = matrix[tag_id, :].sum().item()

        precision = (
            true_positive / predicted_count
            if predicted_count
            else 0.0
        )
        recall = (
            true_positive / actual_count
            if actual_count
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{tag:<8}"
            f"{precision:>11.2%}"
            f"{recall:>12.2%}"
            f"{f1:>12.2%}"
            f"{actual_count:>10}"
        )

if __name__ == "__main__":
    main()