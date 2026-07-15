import argparse
from pathlib import Path

import torch

from vexo.inference import load_pos_model, predict_pos_tags

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict POS tags for pre-tokenized Norwegian text."
    )
    parser.add_argument(
        "tokens",
        nargs="+",
        help="Tokens to analyze.",
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

    model, word_vocabulary, tag_vocabulary = load_pos_model(
        arguments.checkpoint,
        device
    )
    tags = predict_pos_tags(
        model,
        arguments.tokens,
        word_vocabulary,
        tag_vocabulary,
        device
    )

    for token, tag in zip(arguments.tokens, tags):
        print(f"{token}\t{tag}")

if __name__ == "__main__":
    main()