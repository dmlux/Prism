import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from prism.conllu import read_sentences
from prism.data import (
    build_norwegian_schema,
    build_norwegian_ud_lemma_decoder,
    encode_norwegian_sentences,
)
from prism.evaluation.classification import (
    calculate_classification_metrics,
)
from prism.evaluation import (
    TokenFrequencyClass,
    TokenFrequencyProfile,
    UniversalDependenciesEvaluationAccumulator,
    build_universal_dependencies_reference_batch,
    serialize_universal_dependencies_evaluation_metrics,
)
from prism.evaluation.reporting import (
    format_classification_metric_rows,
    format_scalar_metric_rows,
    format_token_slice_metric_rows,
)
from prism.languages import ModelRole
from prism.languages.norwegian import (
    norwegian_model_supports_language_tag,
    norwegian_profile_for_language_tag,
)
from prism.modeling import (
    build_pretrained_token_tagger,
    load_backbone_tokenizer,
)
from prism.schema.serialization import (
    serialize_token_task_schema,
)
from prism.training import (
    backbone_layer_aggregation_strategy_from_checkpoint,
    evaluate_supervised_token_task_epoch,
    iter_supervised_token_task_batches,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
    character_vocabulary_from_checkpoint,
    maximum_character_count_from_checkpoint,
    morphology_logit_correction_from_checkpoint,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineEvaluationArguments:
    checkpoint_path: Path
    analysis_path: Path
    language_tag: str
    device: str
    treebank_release: str
    morphology_logit_correction_strength: float


def parse_evaluation_arguments(
    arguments: Sequence[str] | None = None,
) -> BaselineEvaluationArguments:
    parser = argparse.ArgumentParser(
        description="Evaluate a Norwegian student baseline.",
    )
    parser.add_argument(
        "--language-tag",
        choices=("nb", "nn"),
        default="nb",
    )
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/nb-student-baseline/best.pt"),
        dest="checkpoint_path",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("runs/nb-student-baseline/development-analysis-logit-zero.json"),
        dest="analysis_path",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="mps",
    )
    parser.add_argument(
        "--morphology-logit-correction-strength",
        type=float,
        default=0.0,
        help=(
            "Subtract this fraction of log checkpoint class weights before "
            "morphology decoding (0 disables the evaluation ablation)."
        ),
    )

    parsed_arguments = parser.parse_args(arguments)
    if not 0.0 <= parsed_arguments.morphology_logit_correction_strength <= 1.0:
        parser.error("--morphology-logit-correction-strength must be between 0 and 1")

    return BaselineEvaluationArguments(
        language_tag=parsed_arguments.language_tag,
        checkpoint_path=parsed_arguments.checkpoint_path,
        analysis_path=parsed_arguments.analysis_path,
        device=parsed_arguments.device,
        treebank_release=parsed_arguments.treebank_release,
        morphology_logit_correction_strength=(
            parsed_arguments.morphology_logit_correction_strength
        ),
    )


def main() -> None:
    arguments = parse_evaluation_arguments()
    checkpoint_path = arguments.checkpoint_path
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_token_task_checkpoint_format(checkpoint)
    morphology_logit_correction = morphology_logit_correction_from_checkpoint(
        checkpoint,
        strength=arguments.morphology_logit_correction_strength,
    )

    profile = norwegian_profile_for_language_tag(
        arguments.language_tag,
        treebank_release=arguments.treebank_release,
    )

    checkpoint_treebank_release = checkpoint.get("treebank_release", "current")
    if checkpoint_treebank_release != arguments.treebank_release:
        raise ValueError(
            "Checkpoint treebank release does not match the requested release: "
            f"{checkpoint_treebank_release!r}"
        )

    checkpoint_language_tag = checkpoint.get("language_tag")
    if not isinstance(
        checkpoint_language_tag,
        str,
    ) or not norwegian_model_supports_language_tag(
        checkpoint_language_tag,
        profile.language_tag,
    ):
        raise ValueError(
            "Checkpoint language tag does not support "
            f"the selected profile: {checkpoint_language_tag!r}"
        )

    raw_model_role = checkpoint.get(
        "model_role",
        "student",
    )
    if raw_model_role not in ("student", "teacher"):
        raise ValueError(f"Checkpoint model role is invalid: {raw_model_role!r}")

    model_role = cast(
        ModelRole,
        raw_model_role,
    )

    raw_schema_language_tags = checkpoint.get("schema_language_tags")

    if raw_schema_language_tags is None:
        schema_language_tags = (profile.language_tag,)
    elif isinstance(raw_schema_language_tags, (list, tuple)) and all(
        isinstance(language_tag, str) for language_tag in raw_schema_language_tags
    ):
        schema_language_tags = tuple(raw_schema_language_tags)
    else:
        raise ValueError("Checkpoint schema language tags are invalid.")

    schema_profiles = tuple(
        norwegian_profile_for_language_tag(
            language_tag,
            treebank_release=arguments.treebank_release,
        )
        for language_tag in schema_language_tags
    )

    schema_training_tokens = tuple(
        sentence
        for schema_profile in schema_profiles
        for sentence in read_sentences(schema_profile.gold_treebank.training_path)
    )

    development_tokens = read_sentences(profile.gold_treebank.development_path)

    schema = build_norwegian_schema(schema_training_tokens)

    if checkpoint["schema"] != (serialize_token_task_schema(schema)):
        raise ValueError("Checkpoint schema does not match the pinned training data.")

    backbone_spec = profile.backbone_for_role(model_role)

    if checkpoint["backbone_model_id"] != (backbone_spec.model_id):
        raise ValueError("Checkpoint backbone model does not match.")
    if checkpoint["backbone_revision"] != (backbone_spec.revision):
        raise ValueError("Checkpoint backbone revision does not match.")

    development_corpus = encode_norwegian_sentences(
        development_tokens,
        schema=schema,
    )
    batch_size = int(checkpoint["training_config"]["batch_size"])
    development_sentence_batches = tuple(
        development_corpus.sentences[start : start + batch_size]
        for start in range(
            0,
            len(development_corpus.sentences),
            batch_size,
        )
    )
    development_reference_batches = tuple(
        build_universal_dependencies_reference_batch(
            development_tokens[start : start + batch_size]
        )
        for start in range(
            0,
            len(development_tokens),
            batch_size,
        )
    )
    frequency_profile = TokenFrequencyProfile.from_token_sequences(
        tuple(
            tuple(token.text for token in sentence)
            for sentence in schema_training_tokens
        )
    )
    pretokenized_development_batches = tuple(
        tuple(sentence.model_input for sentence in sentence_batch)
        for sentence_batch in development_sentence_batches
    )
    token_slice_masks = {
        frequency_class.value: frequency_profile.build_batch_masks(
            pretokenized_development_batches,
            frequency_class=frequency_class,
        )
        for frequency_class in (
            TokenFrequencyClass.RARE,
            TokenFrequencyClass.OOV,
        )
    }

    tokenizer = load_backbone_tokenizer(backbone_spec)
    pooling_strategy = token_pooling_strategy_from_checkpoint(checkpoint)
    head_architecture = token_task_head_architecture_from_checkpoint(checkpoint)
    character_vocabulary = character_vocabulary_from_checkpoint(
        checkpoint,
        architecture=head_architecture,
    )
    maximum_character_count = maximum_character_count_from_checkpoint(
        checkpoint,
        architecture=head_architecture,
    )
    layer_aggregation_strategy = backbone_layer_aggregation_strategy_from_checkpoint(
        checkpoint
    )
    model = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
        pooling_strategy=pooling_strategy,
        head_architecture=head_architecture,
        layer_aggregation_strategy=layer_aggregation_strategy,
        character_vocabulary_size=(
            None if character_vocabulary is None else character_vocabulary.size
        ),
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    print(
        "Evaluating checkpoint epoch:",
        int(checkpoint["epoch_index"]) + 1,
    )
    print("Treebank release:", arguments.treebank_release)
    print("Token pooling:", pooling_strategy.value)
    print("Task-head architecture:", head_architecture.value)
    print("Backbone layer aggregation:", layer_aggregation_strategy.value)
    print(
        "Morphology logit correction:",
        f"{arguments.morphology_logit_correction_strength:.2f}",
    )

    metrics = evaluate_supervised_token_task_epoch(
        model=model,
        batches=iter_supervised_token_task_batches(
            tokenizer=tokenizer,
            sentence_batches=(development_sentence_batches),
            character_vocabulary=character_vocabulary,
            maximum_character_count=(
                32 if maximum_character_count is None else maximum_character_count
            ),
        ),
        device=torch.device(arguments.device),
        morphology_schema=schema.morphology,
        token_slice_masks=token_slice_masks,
        universal_dependencies_accumulator=(
            UniversalDependenciesEvaluationAccumulator(
                schema=schema,
                reference_batches=development_reference_batches,
                lemma_decoder=build_norwegian_ud_lemma_decoder(schema_training_tokens),
            )
        ),
        morphology_logit_correction=morphology_logit_correction,
    )

    if metrics.universal_dependencies is None:
        raise RuntimeError("UD-compatible metrics were not calculated.")

    for row in format_scalar_metric_rows(
        metric_names=(
            "Development loss",
            "UPOS accuracy",
            "Lemma-rule accuracy",
            "UD UPOS F1",
            "UD UFeats F1",
            "UD Lemmas F1",
        ),
        values=(
            metrics.losses.total_loss,
            metrics.upos_accuracy,
            metrics.lemma_rule_accuracy,
            metrics.universal_dependencies.upos.f1,
            metrics.universal_dependencies.ufeats.f1,
            metrics.universal_dependencies.lemmas.f1,
        ),
    ):
        print(row)

    print()
    print(
        "Token-frequency slices: normalized with NFC + casefold; "
        f"rare=1..{frequency_profile.rare_max_frequency}, "
        "oov=0 training occurrences."
    )
    for token_slice in metrics.token_slices:
        print()
        for row in format_token_slice_metric_rows(
            slice_name=token_slice.name.upper(),
            metrics=token_slice.metrics,
        ):
            print(row)

    for (
        feature,
        true_positive_counts,
        false_positive_counts,
        false_negative_counts,
        average_precisions,
    ) in zip(
        schema.morphology.features,
        metrics.morphology_true_positive_counts,
        metrics.morphology_false_positive_counts,
        metrics.morphology_false_negative_counts,
        metrics.morphology_average_precisions,
        strict=True,
    ):
        print()
        print(feature.name)

        label_metrics = tuple(
            calculate_classification_metrics(
                true_positive_count=true_positive,
                false_positive_count=false_positive,
                false_negative_count=false_negative,
            )
            for true_positive, false_positive, false_negative in zip(
                true_positive_counts,
                false_positive_counts,
                false_negative_counts,
                strict=True,
            )
        )

        for row in format_classification_metric_rows(
            labels=feature.labels,
            metrics=label_metrics,
            average_precisions=average_precisions,
        ):
            print(row)

    analysis_path = arguments.analysis_path
    analysis_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    serialized_metrics = asdict(metrics)
    serialized_metrics["universal_dependencies"] = (
        serialize_universal_dependencies_evaluation_metrics(
            metrics.universal_dependencies
        )
    )
    analysis_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch_index": int(checkpoint["epoch_index"]),
                "evaluation_policy": {
                    "morphology_logit_correction_strength": (
                        arguments.morphology_logit_correction_strength
                    ),
                    "morphology_logit_correction_weight_source": (
                        None if morphology_logit_correction is None else "checkpoint"
                    ),
                },
                "schema": (serialize_token_task_schema(schema)),
                "metrics": serialized_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Analysis:", analysis_path)


if __name__ == "__main__":
    main()
