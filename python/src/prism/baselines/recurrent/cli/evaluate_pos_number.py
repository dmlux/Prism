import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.conllu import read_sentences
from prism.baselines.recurrent.dataset import (
    CharacterFeatureDataset,
    collate_character_feature_sentences,
)
from prism.baselines.recurrent.inference import load_multitask_model
from prism.baselines.recurrent.training import (
    evaluate_multitask,
    multitask_feature_confusion_matrix,
)
from prism.baselines.recurrent.vocabulary import NO_FEATURE

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
            "pos_number_bilstm.pt"
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
        feature_name,
        feature_values,
    ) = load_multitask_model(
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
        CharacterFeatureDataset(
            sentences,
            words,
            tags,
            characters,
            feature_name,
            feature_values,
        ),
        batch_size=64,
        collate_fn=collate_character_feature_sentences,
    )

    (
        pos_loss,
        pos_accuracy,
        feature_loss,
        feature_accuracy,
        annotated_accuracy,
    ) = evaluate_multitask(
        model,
        loader,
        device,
        no_feature_id=feature_values[NO_FEATURE],
    )

    feature_matrix = multitask_feature_confusion_matrix(
        model,
        loader,
        device,
        len(feature_values),
    )

    print("Split:", arguments.split)
    print("Sätze:", len(sentences))
    print("POS-Loss:", f"{pos_loss:.4f}")
    print("POS-Genauigkeit:", f"{pos_accuracy:.2%}")
    print(
        f"{feature_name}-Loss:",
        f"{feature_loss:.4f}",
    )
    print(
        f"{feature_name}-Genauigkeit:",
        f"{feature_accuracy:.2%}",
    )
    print(
        f"{feature_name} annotiert:",
        f"{annotated_accuracy:.2%}",
    )

    print()
    print(
        f"{feature_name:<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )

    for value, value_id in sorted(
        feature_values.items(),
        key=lambda item: item[1],
    ):
        true_positive = feature_matrix[
            value_id,
            value_id,
        ].item()
        predicted_count = feature_matrix[
            :,
            value_id,
        ].sum().item()
        actual_count = feature_matrix[
            value_id,
            :,
        ].sum().item()

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
            f"{value:<12}"
            f"{precision:>11.2%}"
            f"{recall:>12.2%}"
            f"{f1:>12.2%}"
            f"{actual_count:>10}"
        )

if __name__ == "__main__":
    main()
