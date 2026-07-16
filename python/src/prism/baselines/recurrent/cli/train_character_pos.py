from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism.conllu import read_sentences
from prism.baselines.recurrent.dataset import (
    CharacterPosDataset,
    collate_character_sentences,
)
from prism.baselines.recurrent.model import CharacterBiLSTMPosTagger
from prism.baselines.recurrent.training import (
    evaluate_character,
    train_character_epoch,
)
from prism.baselines.recurrent.vocabulary import (
    build_character_vocabulary,
    build_tag_vocabulary,
    build_word_vocabulary,
)


def main() -> None:
    data_root = Path("data/raw/UD_Norwegian-Bokmaal")

    training = read_sentences(data_root / "no_bokmaal-ud-train.conllu")

    development = read_sentences(data_root / "no_bokmaal-ud-dev.conllu")

    word_vocabulary = build_word_vocabulary(training)
    tag_vocabulary = build_tag_vocabulary(training)
    character_vocabulary = build_character_vocabulary(training)

    training_loader = DataLoader(
        CharacterPosDataset(
            training, word_vocabulary, tag_vocabulary, character_vocabulary
        ),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_character_sentences,
    )

    development_loader = DataLoader(
        CharacterPosDataset(
            development,
            word_vocabulary,
            tag_vocabulary,
            character_vocabulary,
        ),
        batch_size=64,
        collate_fn=collate_character_sentences,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(42)

    model = CharacterBiLSTMPosTagger(
        vocabulary_size=len(word_vocabulary),
        character_count=len(character_vocabulary),
        tag_count=len(tag_vocabulary),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_directory = Path("models/norwegian-bokmaal")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / "pos_character_bilstm_10_epochs.pt"

    best_accuracy = 0.0

    for epoch in range(1, 11):
        training_loss = train_character_epoch(
            model,
            training_loader,
            optimizer,
            device,
        )

        development_loss, development_accuracy = evaluate_character(
            model,
            development_loader,
            device,
        )

        print(
            f"Epoche {epoch}: "
            f"Training-Loss={training_loss:.4f}, "
            f"Entwicklungs-Loss={development_loss:.4f}, "
            f"Genauigkeit={development_accuracy:.2%}"
        )

        if development_accuracy > best_accuracy:
            best_accuracy = development_accuracy

            torch.save(
                {
                    "model_type": "character_bilstm_pos",
                    "model_state": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "word_vocabulary": word_vocabulary,
                    "character_vocabulary": character_vocabulary,
                    "tag_vocabulary": tag_vocabulary,
                    "word_embedding_size": 64,
                    "character_embedding_size": 32,
                    "character_hidden_size": 32,
                    "hidden_size": 128,
                    "epoch": epoch,
                    "development_accuracy": development_accuracy,
                },
                checkpoint_path,
            )

            print("Bestes Modell gespeichert:", checkpoint_path)

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
