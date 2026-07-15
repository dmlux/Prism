import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.conllu import read_sentences
from prism.baselines.recurrent.dataset import PosDataset, collate_sentences
from prism.baselines.recurrent.inference import load_pos_model
from prism.baselines.recurrent.training import (
    evaluate,
    evaluate_knownness,
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
            "models/norwegian-bokmaal/pos_bilstm.pt"
        ),
    )
    arguments = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, words, tags = load_pos_model(
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
        PosDataset(sentences, words, tags),
        batch_size=64,
        collate_fn=collate_sentences,
    )

    loss, accuracy = evaluate(model, loader, device)

    (
        known_correct,
        known_count,
        unknown_correct,
        unknown_count,
    ) = evaluate_knownness(
        model,
        loader,
        device,
        words["<UNK>"],
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

if __name__ == "__main__":
    main()
