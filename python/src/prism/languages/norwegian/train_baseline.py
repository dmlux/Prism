import argparse
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from prism.conllu import read_sentences
from prism.data import (
    build_norwegian_schema,
    encode_norwegian_sentences,
)
from prism.evaluation.reporting import (
    format_morphology_accuracy_rows,
    format_scalar_metric_rows,
)
from prism.languages import ModelRole
from prism.languages.norwegian import (
    NORWEGIAN_WRITTEN_STANDARD_PROFILES,
    norwegian_model_supports_language_tag,
    norwegian_training_profiles_for_language_tag,
)
from prism.modeling import (
    BackboneLayerAggregationStrategy,
    PretrainedBackboneSpec,
    TokenPoolingStrategy,
    TokenTagger,
    TokenTaskHeadArchitecture,
    build_pretrained_token_tagger,
    load_backbone_tokenizer,
)
from prism.schema import (
    CharacterVocabularySchema,
    TokenTaskSchema,
    build_character_vocabulary_schema,
)
from prism.schema.serialization import (
    serialize_character_vocabulary_schema,
    serialize_token_task_schema,
)
from prism.training import (
    TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
    DistilledEpochMetrics,
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
    train_distilled_token_task_epoch,
    token_pooling_strategy_from_checkpoint,
    backbone_layer_aggregation_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
    character_vocabulary_from_checkpoint,
)


CHARACTER_MAXIMUM_COUNT = 32


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineTrainingArguments:
    checkpoint_path: Path
    morphology_weight_cap: float | None
    language_tag: str
    model_role: ModelRole
    teacher_checkpoint_path: Path | None
    distillation_temperature: float
    distillation_weight: float
    token_pooling_strategy: TokenPoolingStrategy
    token_task_head_architecture: TokenTaskHeadArchitecture
    backbone_layer_aggregation: BackboneLayerAggregationStrategy
    epoch_count: int


def parse_training_arguments(
    arguments: Sequence[str] | None = None,
) -> BaselineTrainingArguments:
    parser = argparse.ArgumentParser(
        description="Train a Norwegian student baseline.",
    )
    parser.add_argument(
        "--language-tag",
        choices=("nb", "nn", "no"),
        default="nb",
    )
    parser.add_argument(
        "--model-role",
        choices=("student", "teacher"),
        default="student",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/nb-student-baseline/best.pt"),
        dest="checkpoint_path",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        dest="teacher_checkpoint_path",
    )
    parser.add_argument(
        "--distillation-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--distillation-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--token-pooling",
        choices=tuple(strategy.value for strategy in TokenPoolingStrategy),
        default=TokenPoolingStrategy.MEAN.value,
    )
    parser.add_argument(
        "--task-head-architecture",
        choices=tuple(architecture.value for architecture in TokenTaskHeadArchitecture),
        default=(
            TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN.value
        ),
    )
    parser.add_argument(
        "--backbone-layer-aggregation",
        choices=tuple(strategy.value for strategy in BackboneLayerAggregationStrategy),
        default=BackboneLayerAggregationStrategy.LEARNED_LAST_FOUR.value,
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--morphology-weight-cap",
        "--morphology-positive-weight-cap",
        type=float,
        default=None,
        dest="morphology_weight_cap",
    )

    parsed_arguments = parser.parse_args(arguments)

    if parsed_arguments.distillation_temperature <= 0.0:
        parser.error("--distillation-temperature must be greater than zero")
    if parsed_arguments.distillation_weight < 0.0:
        parser.error("--distillation-weight must be non-negative")
    if parsed_arguments.epoch_count <= 0:
        parser.error("--epoch-count must be greater than zero")
    if (
        parsed_arguments.teacher_checkpoint_path is not None
        and parsed_arguments.model_role != "student"
    ):
        parser.error("--teacher-checkpoint can only be used while training a student")

    return BaselineTrainingArguments(
        language_tag=parsed_arguments.language_tag,
        checkpoint_path=parsed_arguments.checkpoint_path,
        morphology_weight_cap=parsed_arguments.morphology_weight_cap,
        model_role=cast(
            ModelRole,
            parsed_arguments.model_role,
        ),
        teacher_checkpoint_path=parsed_arguments.teacher_checkpoint_path,
        distillation_temperature=parsed_arguments.distillation_temperature,
        distillation_weight=parsed_arguments.distillation_weight,
        token_pooling_strategy=TokenPoolingStrategy(parsed_arguments.token_pooling),
        token_task_head_architecture=TokenTaskHeadArchitecture(
            parsed_arguments.task_head_architecture
        ),
        backbone_layer_aggregation=BackboneLayerAggregationStrategy(
            parsed_arguments.backbone_layer_aggregation
        ),
        epoch_count=parsed_arguments.epoch_count,
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


def _load_distillation_teacher(
    *,
    checkpoint_path: Path | None,
    backbone_spec: PretrainedBackboneSpec,
    schema: TokenTaskSchema,
    requested_language_tag: str,
    student_character_vocabulary: CharacterVocabularySchema | None = None,
) -> TokenTagger | None:
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_token_task_checkpoint_format(checkpoint)

    checkpoint_language_tag = checkpoint.get("language_tag")
    if not isinstance(
        checkpoint_language_tag,
        str,
    ) or not norwegian_model_supports_language_tag(
        checkpoint_language_tag,
        requested_language_tag,
    ):
        raise ValueError(
            "Teacher checkpoint does not support "
            f"language tag {requested_language_tag!r}."
        )

    if checkpoint.get("model_role") != "teacher":
        raise ValueError("Distillation requires a teacher checkpoint.")

    if checkpoint.get("schema") != serialize_token_task_schema(schema):
        raise ValueError(
            "Teacher checkpoint schema does not match the student training schema."
        )

    if checkpoint.get("backbone_model_id") != backbone_spec.model_id:
        raise ValueError("Teacher checkpoint backbone model does not match.")

    if checkpoint.get("backbone_revision") != backbone_spec.revision:
        raise ValueError("Teacher checkpoint backbone revision does not match.")

    teacher_architecture = token_task_head_architecture_from_checkpoint(checkpoint)
    teacher_character_vocabulary = character_vocabulary_from_checkpoint(
        checkpoint,
        architecture=teacher_architecture,
    )
    if teacher_character_vocabulary != student_character_vocabulary and (
        teacher_character_vocabulary is not None
    ):
        raise ValueError("Character-aware teacher and student vocabularies must match.")

    teacher = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
        pooling_strategy=token_pooling_strategy_from_checkpoint(checkpoint),
        head_architecture=teacher_architecture,
        layer_aggregation_strategy=(
            backbone_layer_aggregation_strategy_from_checkpoint(checkpoint)
        ),
        character_vocabulary_size=(
            None
            if teacher_character_vocabulary is None
            else teacher_character_vocabulary.size
        ),
    )
    teacher.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    teacher.requires_grad_(False)
    teacher.eval()

    return teacher


def main() -> None:
    arguments = parse_training_arguments()
    training_profiles = norwegian_training_profiles_for_language_tag(
        arguments.language_tag
    )

    training_tokens = tuple(
        sentence
        for training_profile in training_profiles
        for sentence in read_sentences(training_profile.gold_treebank.training_path)
    )
    development_tokens = tuple(
        sentence
        for training_profile in training_profiles
        for sentence in read_sentences(training_profile.gold_treebank.development_path)
    )

    schema_training_tokens = tuple(
        sentence
        for written_standard_profile in (NORWEGIAN_WRITTEN_STANDARD_PROFILES)
        for sentence in read_sentences(
            written_standard_profile.gold_treebank.training_path
        )
    )

    print("Model role:", arguments.model_role)
    print("Token pooling:", arguments.token_pooling_strategy.value)
    print("Task-head architecture:", arguments.token_task_head_architecture.value)
    print("Backbone layer aggregation:", arguments.backbone_layer_aggregation.value)
    print("Training sentences:", len(training_tokens))
    print(
        "Development sentences:",
        len(development_tokens),
    )

    schema = build_norwegian_schema(schema_training_tokens)
    character_vocabulary = (
        build_character_vocabulary_schema(
            tokens=(token.text for sentence in training_tokens for token in sentence),
        )
        if arguments.token_task_head_architecture.uses_character_encoder
        else None
    )
    training_corpus = encode_norwegian_sentences(
        training_tokens,
        schema=schema,
    )
    development_corpus = encode_norwegian_sentences(
        development_tokens,
        schema=schema,
    )

    config = SupervisedTrainingConfig(
        epoch_count=arguments.epoch_count,
        batch_size=16,
        backbone_learning_rate=2e-5,
        task_head_learning_rate=5e-4,
        weight_decay=0.01,
        max_gradient_norm=1.0,
        warmup_ratio=0.1,
        random_seed=42,
        morphology_weight_cap=arguments.morphology_weight_cap,
    )

    torch.manual_seed(config.random_seed)
    device = torch.device("mps")

    loss_weights = build_token_task_loss_weights(
        targets=tuple(
            target
            for sentence in training_corpus.sentences
            for target in sentence.targets
        ),
        morphology_schema=schema.morphology,
        config=config,
    )

    if loss_weights is not None:
        loss_weights = loss_weights.to(device)

    backbone_spec = training_profiles[0].backbone_for_role(arguments.model_role)
    tokenizer = load_backbone_tokenizer(backbone_spec)
    model = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
        pooling_strategy=arguments.token_pooling_strategy,
        head_architecture=arguments.token_task_head_architecture,
        layer_aggregation_strategy=arguments.backbone_layer_aggregation,
        character_vocabulary_size=(
            None if character_vocabulary is None else character_vocabulary.size
        ),
    )

    teacher = _load_distillation_teacher(
        checkpoint_path=arguments.teacher_checkpoint_path,
        backbone_spec=training_profiles[0].teacher_backbone,
        schema=schema,
        requested_language_tag=arguments.language_tag,
        student_character_vocabulary=character_vocabulary,
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
        task_feature_extractor=model.layer_aggregation,
        task_input_encoder=model.character_encoder,
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
    ) -> SupervisedEpochMetrics | DistilledEpochMetrics:
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

        batches = _report_progress(
            iter_supervised_token_task_batches(
                tokenizer=tokenizer,
                sentence_batches=sentence_batches,
                character_vocabulary=character_vocabulary,
                maximum_character_count=CHARACTER_MAXIMUM_COUNT,
            ),
            label="Training",
            total=len(sentence_batches),
        )

        if teacher is None:
            return train_supervised_token_task_epoch(
                model=model,
                batches=batches,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                max_gradient_norm=config.max_gradient_norm,
                morphology_schema=schema.morphology,
                loss_weights=loss_weights,
            )

        return train_distilled_token_task_epoch(
            student=model,
            teacher=teacher,
            batches=batches,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_gradient_norm=config.max_gradient_norm,
            temperature=arguments.distillation_temperature,
            distillation_weight=arguments.distillation_weight,
            morphology_schema=schema.morphology,
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
                    character_vocabulary=character_vocabulary,
                    maximum_character_count=CHARACTER_MAXIMUM_COUNT,
                ),
                label="Development",
                total=len(development_sentence_batches),
            ),
            device=device,
            morphology_schema=schema.morphology,
        )

        for row in format_scalar_metric_rows(
            metric_names=(
                "Development total loss",
                "Development UPOS accuracy",
                "Development lemma-rule accuracy",
            ),
            values=(
                metrics.losses.total_loss,
                metrics.upos_accuracy,
                metrics.lemma_rule_accuracy,
            ),
        ):
            print(row)

        for row in format_morphology_accuracy_rows(
            feature_names=tuple(feature.name for feature in schema.morphology.features),
            overall_accuracies=metrics.morphology_accuracies,
            annotated_accuracies=metrics.morphology_annotated_accuracies,
            prefix="Development",
        ):
            print(row)

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
                "checkpoint_format_version": TOKEN_TASK_CHECKPOINT_FORMAT_VERSION,
                "epoch_index": epoch.epoch_index,
                "language_tag": arguments.language_tag,
                "model_role": arguments.model_role,
                "token_pooling_strategy": arguments.token_pooling_strategy.value,
                "token_task_head_architecture": (
                    arguments.token_task_head_architecture.value
                ),
                "backbone_layer_aggregation": (
                    arguments.backbone_layer_aggregation.value
                ),
                "character_vocabulary": (
                    None
                    if character_vocabulary is None
                    else serialize_character_vocabulary_schema(character_vocabulary)
                ),
                "maximum_character_count": (
                    None if character_vocabulary is None else CHARACTER_MAXIMUM_COUNT
                ),
                "teacher_checkpoint_path": (
                    None
                    if arguments.teacher_checkpoint_path is None
                    else str(arguments.teacher_checkpoint_path)
                ),
                "teacher_backbone_model_id": (
                    None
                    if teacher is None
                    else training_profiles[0].teacher_backbone.model_id
                ),
                "teacher_backbone_revision": (
                    None
                    if teacher is None
                    else training_profiles[0].teacher_backbone.revision
                ),
                "distillation_temperature": (
                    None if teacher is None else arguments.distillation_temperature
                ),
                "distillation_weight": (
                    None if teacher is None else arguments.distillation_weight
                ),
                "schema_language_tags": tuple(
                    schema_profile.language_tag
                    for schema_profile in NORWEGIAN_WRITTEN_STANDARD_PROFILES
                ),
                "backbone_model_id": (backbone_spec.model_id),
                "backbone_revision": (backbone_spec.revision),
                "training_config": asdict(config),
                "morphology_weights": (
                    None
                    if loss_weights is None
                    else tuple(
                        weights.detach().cpu().tolist()
                        for weights in loss_weights.morphology_weights
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
