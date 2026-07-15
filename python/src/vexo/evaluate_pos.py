import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vexo.conllu import read_sentences
from vexo.dataset import PosDataset, collate_sentences
from vexo.inference import load_pos_model
from vexo.training import evaluate

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

    print("Split:", arguments.split)
    print("Sätze:", len(sentences))
    print("Loss:", f"{loss:.4f}")
    print("Genauigkeit:", f"{accuracy:.2%}")

if __name__ == "__main__":
    main()