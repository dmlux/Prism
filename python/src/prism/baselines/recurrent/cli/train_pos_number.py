from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.conllu import read_sentences
from prism.baselines.recurrent.dataset import (
    CharacterFeatureDataset,
    collate_character_feature_sentences,
)
from prism.baselines.recurrent.model import CharacterBiLSTMMultiTaskTagger
from prism.baselines.recurrent.training import (
    evaluate_multitask,
    train_multitask_epoch,
)
from prism.baselines.recurrent.vocabulary import (
    NO_FEATURE,
    build_character_vocabulary,
    build_feature_vocabulary,
    build_tag_vocabulary,
    build_word_vocabulary,
)


def main() -> None:
    data_root = Path("data/raw/UD_Norwegian-Bokmaal")

    training = read_sentences(data_root / "no_bokmaal-ud-train.conllu")

    development = read_sentences(data_root / "no_bokmaal-ud-dev.conllu")

    feature_name = "Number"
    feature_vocabulary = build_feature_vocabulary(training, feature_name)

    word_vocabulary = build_word_vocabulary(training)
    tag_vocabulary = build_tag_vocabulary(training)
    character_vocabulary = build_character_vocabulary(training)

    training_loader = DataLoader(
        CharacterFeatureDataset(
            training,
            word_vocabulary,
            tag_vocabulary,
            character_vocabulary,
            feature_name,
            feature_vocabulary,
        ),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_character_feature_sentences,
    )

    development_loader = DataLoader(
        CharacterFeatureDataset(
            development,
            word_vocabulary,
            tag_vocabulary,
            character_vocabulary,
            feature_name,
            feature_vocabulary,
        ),
        batch_size=64,
        collate_fn=collate_character_feature_sentences,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(42)

    model = CharacterBiLSTMMultiTaskTagger(
        vocabulary_size=len(word_vocabulary),
        character_count=len(character_vocabulary),
        tag_count=len(tag_vocabulary),
        feature_count=len(feature_vocabulary),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_directory = Path("models/norwegian-bokmaal")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / "pos_number_bilstm.pt"

    best_development_loss = float("inf")

    for epoch in range(1, 11):
        (
            training_loss,
            training_pos_loss,
            training_feature_loss,
        ) = train_multitask_epoch(
            model,
            training_loader,
            optimizer,
            device,
        )

        (
            development_pos_loss,
            development_pos_accuracy,
            development_feature_loss,
            development_feature_accuracy,
            development_annotated_accuracy,
        ) = evaluate_multitask(
            model,
            development_loader,
            device,
            no_feature_id=feature_vocabulary[NO_FEATURE],
        )

        development_loss = development_pos_loss + development_feature_loss

        print(
            f"Epoche {epoch}: "
            f"Training-Loss={training_loss:.4f}, "
            f"POS-Loss={development_pos_loss:.4f}, "
            f"POS={development_pos_accuracy:.2%}, "
            f"Number-Loss={development_feature_loss:.4f}, "
            f"Number={development_feature_accuracy:.2%}, "
            f"Number annotiert="
            f"{development_annotated_accuracy:.2%}"
        )

        if development_loss < best_development_loss:
            best_development_loss = development_loss

            torch.save(
                {
                    "model_type": "character_bilstm_pos_feature",
                    "model_state": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "word_vocabulary": word_vocabulary,
                    "character_vocabulary": character_vocabulary,
                    "tag_vocabulary": tag_vocabulary,
                    "feature_name": feature_name,
                    "feature_vocabulary": feature_vocabulary,
                    "word_embedding_size": 64,
                    "character_embedding_size": 32,
                    "character_hidden_size": 32,
                    "hidden_size": 128,
                    "feature_loss_weight": 1.0,
                    "epoch": epoch,
                    "development_loss": development_loss,
                    "development_pos_accuracy": (development_pos_accuracy),
                    "development_feature_accuracy": (development_feature_accuracy),
                    "development_annotated_accuracy": (development_annotated_accuracy),
                },
                checkpoint_path,
            )

            print(
                "Bestes gemeinsames Modell gespeichert:",
                checkpoint_path,
            )

    print("Trainingssätze:", len(training))
    print("Trainings-Batches:", len(training_loader))
    print("Entwicklungssätze:", len(development))
    print("Entwicklungs-Batches:", len(development_loader))
    print("Wortverzeichnis:", len(word_vocabulary))
    print("Zeichenverzeichnis:", len(character_vocabulary))
    print("POS-Klassen:", len(tag_vocabulary))
    print("Gerät:", device)


if __name__ == "__main__":
    main()
