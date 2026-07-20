import argparse
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prism.conllu import read_sentences
from prism.data import (
    build_norwegian_schema,
    encode_norwegian_sentences,
)
from prism.languages.norwegian import (
    NORWEGIAN_WRITTEN_STANDARD_PROFILES,
    norwegian_profile_for_language_tag,
)
from prism.modeling import (
    build_pretrained_token_tagger,
    load_backbone_tokenizer,
)
from prism.schema.serialization import serialize_token_task_schema
from prism.training import (
    SupervisedEpochMetrics,
    SupervisedEvaluationMetrics,
    SupervisedTokenTaskBatch,
    SupervisedTrainingConfig,
    SupervisedTrainingEpochResult,
    build_linear_warmup_decay_scheduler,
    build_supervised_adamw_optimizer,
    build_supervised_sentence_batches,
    build_token_task_loss_weights,
    evaluate_supervised_token_task_epoch,
    iter_supervised_token_task_batches,
    run_supervised_training_epochs,
    train_supervised_token_task_epoch,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineTrainingArguments:
    checkpoint_path: Path
    morphology_positive_weight_cap: float | None
    language_tag: str


def parse_training_arguments(
    arguments: Sequence[str] | None = None,
) -> BaselineTrainingArguments:
    parser = argparse.ArgumentParser(
        description="Train a Norwegian student baseline.",
    )
    parser.add_argument(
        "--language-tag",
        choices=("nb", "nn"),
        default="nb",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/nb-student-baseline/best.pt"),
        dest="checkpoint_path",
    )
    parser.add_argument(
        "--morphology-positive-weight-cap",
        type=float,
        default=None,
    )

    parsed_arguments = parser.parse_args(arguments)

    return BaselineTrainingArguments(
        language_tag=parsed_arguments.language_tag,
        checkpoint_path=parsed_arguments.checkpoint_path,
        morphology_positive_weight_cap=(
            parsed_arguments.morphology_positive_weight_cap
        ),
    )


def _report_progress(
    batches: Iterable[SupervisedTokenTaskBatch],
    *,
    label: str,
    total: int,
) -> Iterator[SupervisedTokenTaskBatch]:
    for index, batch in enumerate(batches, start=1):
        if index == 1 or index % 100 == 0 or index == total:
            print(
                f"{label}: Batch {index}/{total}",
                flush=True,
            )

        yield batch


def main() -> None:
    arguments = parse_training_arguments()
    profile = norwegian_profile_for_language_tag(arguments.language_tag)
    treebank = profile.gold_treebank

    training_tokens = read_sentences(treebank.training_path)
    development_tokens = read_sentences(treebank.development_path)

    schema_training_tokens = tuple(
        sentence
        for written_standard_profile in (NORWEGIAN_WRITTEN_STANDARD_PROFILES)
        for sentence in read_sentences(
            written_standard_profile.gold_treebank.training_path
        )
    )

    print("Training sentences:", len(training_tokens))
    print(
        "Development sentences:",
        len(development_tokens),
    )

    schema = build_norwegian_schema(schema_training_tokens)
    training_corpus = encode_norwegian_sentences(
        training_tokens,
        schema=schema,
    )
    development_corpus = encode_norwegian_sentences(
        development_tokens,
        schema=schema,
    )

    config = SupervisedTrainingConfig(
        epoch_count=5,
        batch_size=16,
        backbone_learning_rate=2e-5,
        task_head_learning_rate=5e-4,
        weight_decay=0.01,
        max_gradient_norm=1.0,
        warmup_ratio=0.1,
        random_seed=42,
        morphology_positive_weight_cap=(arguments.morphology_positive_weight_cap),
    )

    torch.manual_seed(config.random_seed)
    device = torch.device("mps")

    loss_weights = build_token_task_loss_weights(
        targets=tuple(
            target
            for sentence in training_corpus.sentences
            for target in sentence.targets
        ),
        config=config,
    )

    if loss_weights is not None:
        loss_weights = loss_weights.to(device)

    backbone_spec = profile.student_backbone
    tokenizer = load_backbone_tokenizer(backbone_spec)
    model = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
    )

    training_batch_count = (
        len(training_corpus.sentences) + config.batch_size - 1
    ) // config.batch_size

    development_sentence_batches = tuple(
        development_corpus.sentences[start : start + config.batch_size]
        for start in range(
            0,
            len(development_corpus.sentences),
            config.batch_size,
        )
    )

    optimizer = build_supervised_adamw_optimizer(
        backbone=model.backbone,
        task_heads=model.heads,
        config=config,
    )
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_step_count=(training_batch_count * config.epoch_count),
        warmup_ratio=config.warmup_ratio,
    )

    checkpoint_path = arguments.checkpoint_path
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def train_epoch(
        epoch_index: int,
    ) -> SupervisedEpochMetrics:
        sentence_batches = build_supervised_sentence_batches(
            sentences=training_corpus.sentences,
            batch_size=config.batch_size,
            random_seed=config.random_seed,
            epoch_index=epoch_index,
        )

        print()
        print(
            f"Epoch {epoch_index + 1}/{config.epoch_count}: training",
            flush=True,
        )

        return train_supervised_token_task_epoch(
            model=model,
            batches=_report_progress(
                iter_supervised_token_task_batches(
                    tokenizer=tokenizer,
                    sentence_batches=sentence_batches,
                ),
                label="Training",
                total=len(sentence_batches),
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_gradient_norm=(config.max_gradient_norm),
            loss_weights=loss_weights,
        )

    def evaluate_epoch(
        epoch_index: int,
    ) -> SupervisedEvaluationMetrics:
        print(
            f"Epoch {epoch_index + 1}/{config.epoch_count}: development",
            flush=True,
        )

        metrics = evaluate_supervised_token_task_epoch(
            model=model,
            batches=_report_progress(
                iter_supervised_token_task_batches(
                    tokenizer=tokenizer,
                    sentence_batches=(development_sentence_batches),
                ),
                label="Development",
                total=len(development_sentence_batches),
            ),
            device=device,
            morphology_schema=schema.morphology,
        )

        print(
            "Development total loss:",
            metrics.losses.total_loss,
        )
        print(
            "Development UPOS accuracy:",
            metrics.upos_accuracy,
        )
        print(
            "Development lemma-rule accuracy:",
            metrics.lemma_rule_accuracy,
        )

        for feature, overall, annotated in zip(
            schema.morphology.features,
            metrics.morphology_accuracies,
            metrics.morphology_annotated_accuracies,
            strict=True,
        ):
            print(
                f"Development {feature.name}: overall={overall}, annotated={annotated}"
            )

        return metrics

    def save_new_best(
        epoch: SupervisedTrainingEpochResult,
    ) -> None:
        model_state_dict = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        temporary_path = checkpoint_path.with_suffix(".tmp")

        torch.save(
            {
                "checkpoint_format_version": 2,
                "epoch_index": epoch.epoch_index,
                "language_tag": profile.language_tag,
                "schema_language_tags": tuple(
                    schema_profile.language_tag
                    for schema_profile in NORWEGIAN_WRITTEN_STANDARD_PROFILES
                ),
                "backbone_model_id": (backbone_spec.model_id),
                "backbone_revision": (backbone_spec.revision),
                "training_config": asdict(config),
                "morphology_positive_weights": (
                    None
                    if loss_weights is None
                    else tuple(
                        weights.detach().cpu().tolist()
                        for weights in (loss_weights.morphology_positive_weights)
                    )
                ),
                "schema": serialize_token_task_schema(schema),
                "training_metrics": asdict(epoch.training_metrics),
                "development_metrics": asdict(epoch.development_metrics),
                "model_state_dict": model_state_dict,
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)

        print(
            "Saved new best checkpoint from epoch",
            epoch.epoch_index + 1,
        )

    run_result = run_supervised_training_epochs(
        epoch_count=config.epoch_count,
        train_epoch=train_epoch,
        evaluate_epoch=evaluate_epoch,
        on_new_best=save_new_best,
    )

    print()
    print(
        "Best epoch:",
        run_result.best_epoch_index + 1,
    )
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
