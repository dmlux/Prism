import argparse
from pathlib import Path

import torch

from vexo.inference import (
    load_multitask_model,
    predict_multitask,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict POS and morphology for "
            "pre-tokenized Norwegian text."
        )
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
            "models/norwegian-bokmaal/"
            "pos_number_bilstm.pt"
        ),
    )
    arguments = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    (
        model,
        word_vocabulary,
        character_vocabulary,
        tag_vocabulary,
        feature_name,
        feature_vocabulary,
    ) = load_multitask_model(
        arguments.checkpoint,
        device,
    )

    predictions = predict_multitask(
        model,
        arguments.tokens,
        word_vocabulary,
        character_vocabulary,
        tag_vocabulary,
        feature_vocabulary,
        device,
    )

    print(f"Token\tPOS\t{feature_name}")

    for token, (tag, feature_value) in zip(
        arguments.tokens,
        predictions,
    ):
        print(
            f"{token}\t{tag}\t{feature_value}"
        )

if __name__ == "__main__":
    main()