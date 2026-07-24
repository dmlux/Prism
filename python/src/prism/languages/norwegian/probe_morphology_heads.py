import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import cast

import torch

from prism.conllu import read_sentences
from prism.data import build_norwegian_schema, encode_norwegian_sentences
from prism.evaluation import TokenFrequencyClass, TokenFrequencyProfile
from prism.languages import ModelRole
from prism.languages.norwegian import (
    norwegian_model_supports_language_tag,
    norwegian_profile_for_language_tag,
    norwegian_training_profiles_for_language_tag,
)
from prism.modeling import build_pretrained_token_tagger, load_backbone_tokenizer
from prism.schema.serialization import serialize_token_task_schema
from prism.training import (
    MorphologyHeadProbeArchitecture,
    MorphologyHeadProbeConfig,
    MorphologyProbeAccuracy,
    MorphologyProbeDataset,
    MorphologyProbeFeatureMetrics,
    backbone_layer_aggregation_strategy_from_checkpoint,
    character_vocabulary_from_checkpoint,
    deserialize_morphology_probe_dataset,
    evaluate_morphology_head_probe,
    extract_morphology_probe_dataset,
    iter_supervised_token_task_batches,
    maximum_character_count_from_checkpoint,
    morphology_agreement_refiner_spec_from_checkpoint,
    morphology_bundle_reranker_spec_from_checkpoint,
    morphology_pre_head_architecture_from_checkpoint,
    morphology_weights_from_checkpoint,
    serialize_morphology_probe_dataset,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    train_morphology_head_probe,
    validate_token_task_checkpoint_format,
)


MORPHOLOGY_PROBE_CACHE_FORMAT_VERSION = 1
MORPHOLOGY_PROBE_REPORT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyProbeArguments:
    checkpoint_path: Path
    analysis_path: Path
    representation_cache_path: Path | None
    rebuild_representation_cache: bool
    evaluation_language_tags: tuple[str, ...]
    treebank_release: str
    device: str
    architectures: tuple[MorphologyHeadProbeArchitecture, ...]
    config: MorphologyHeadProbeConfig
    morphology_logit_correction_strength: float


def parse_probe_arguments(
    arguments: Sequence[str] | None = None,
) -> MorphologyProbeArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Train small morphology heads on representations extracted from "
            "a completely frozen Norwegian checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("runs/morphology-head-probe/analysis.json"),
        dest="analysis_path",
    )
    parser.add_argument(
        "--representation-cache",
        type=Path,
        dest="representation_cache_path",
        help=(
            "Optional reusable tensor cache. It is validated against the "
            "checkpoint SHA-256, schema, release, and evaluation profiles."
        ),
    )
    parser.add_argument(
        "--rebuild-representation-cache",
        action="store_true",
        help="Ignore and replace an existing representation cache.",
    )
    parser.add_argument(
        "--evaluation-language-tag",
        choices=("nb", "nn"),
        action="append",
        dest="evaluation_language_tags",
        help=(
            "Development profile to evaluate. Repeat for both standards. "
            "The default evaluates every profile supported by the checkpoint."
        ),
    )
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="mps",
    )
    parser.add_argument(
        "--architecture",
        choices=tuple(
            architecture.value for architecture in MorphologyHeadProbeArchitecture
        ),
        action="append",
        dest="architectures",
        help=(
            "Probe architecture. Repeat to select several. The default runs "
            "linear, shared-mlp, and feature-mlp controls."
        ),
    )
    parser.add_argument("--epoch-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout-probability", type=float, default=0.1)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--morphology-logit-correction-strength",
        type=float,
        default=1.0,
        help=(
            "Apply the checkpoint's stored morphology-weight correction to "
            "every probe before decoding."
        ),
    )

    parsed = parser.parse_args(arguments)
    if parsed.rebuild_representation_cache and (
        parsed.representation_cache_path is None
    ):
        parser.error("--rebuild-representation-cache requires --representation-cache")
    if not 0.0 <= parsed.morphology_logit_correction_strength <= 1.0:
        parser.error("--morphology-logit-correction-strength must be between 0 and 1")

    try:
        config = MorphologyHeadProbeConfig(
            epoch_count=parsed.epoch_count,
            batch_size=parsed.batch_size,
            learning_rate=parsed.learning_rate,
            weight_decay=parsed.weight_decay,
            dropout_probability=parsed.dropout_probability,
            max_gradient_norm=parsed.max_gradient_norm,
            random_seed=parsed.random_seed,
        )
    except ValueError as error:
        parser.error(str(error))

    architectures = (
        tuple(
            MorphologyHeadProbeArchitecture(architecture)
            for architecture in parsed.architectures
        )
        if parsed.architectures
        else tuple(MorphologyHeadProbeArchitecture)
    )
    if len(set(architectures)) != len(architectures):
        parser.error("--architecture values must be unique")
    evaluation_language_tags = tuple(parsed.evaluation_language_tags or ())
    if len(set(evaluation_language_tags)) != len(evaluation_language_tags):
        parser.error("--evaluation-language-tag values must be unique")

    return MorphologyProbeArguments(
        checkpoint_path=parsed.checkpoint,
        analysis_path=parsed.analysis_path,
        representation_cache_path=parsed.representation_cache_path,
        rebuild_representation_cache=parsed.rebuild_representation_cache,
        evaluation_language_tags=evaluation_language_tags,
        treebank_release=parsed.treebank_release,
        device=parsed.device,
        architectures=architectures,
        config=config,
        morphology_logit_correction_strength=(
            parsed.morphology_logit_correction_strength
        ),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sentence_batches(
    sentences: Sequence[object],
    *,
    batch_size: int,
) -> tuple[Sequence[object], ...]:
    return tuple(
        sentences[start : start + batch_size]
        for start in range(0, len(sentences), batch_size)
    )


def _progress_callback(
    *,
    label: str,
    total: int,
):
    def report(batch_index: int) -> None:
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == total:
            print(f"{label}: batch {batch_index}/{total}", flush=True)

    return report


def _serialize_accuracy(
    metric: MorphologyProbeAccuracy,
) -> dict[str, int | float | None]:
    return {
        "correct_count": metric.correct_count,
        "token_count": metric.token_count,
        "accuracy": metric.accuracy,
    }


def _serialize_feature_metrics(
    metrics: MorphologyProbeFeatureMetrics,
) -> dict[str, object]:
    return {
        "feature_name": metrics.feature_name,
        "overall": _serialize_accuracy(metrics.overall),
        "annotated": _serialize_accuracy(metrics.annotated),
        "slices": {
            token_slice.name: _serialize_accuracy(token_slice.accuracy)
            for token_slice in metrics.slices
        },
    }


def _format_accuracy(metric: MorphologyProbeAccuracy) -> str:
    return "       -" if metric.accuracy is None else f"{metric.accuracy:8.6f}"


def _print_feature_metrics(
    *,
    language_tag: str,
    architecture: MorphologyHeadProbeArchitecture,
    metrics: tuple[MorphologyProbeFeatureMetrics, ...],
) -> None:
    feature_width = max(len(metric.feature_name) for metric in metrics)
    print()
    print(f"{language_tag}: {architecture.value}")
    print(
        f"{'Feature':<{feature_width}}    "
        f"{'Overall':>8}    {'Annotated':>9}    {'Rare':>8}    {'OOV':>8}"
    )
    for feature in metrics:
        slices = {
            token_slice.name: token_slice.accuracy for token_slice in feature.slices
        }
        print(
            f"{feature.feature_name:<{feature_width}}    "
            f"{_format_accuracy(feature.overall)}    "
            f"{_format_accuracy(feature.annotated):>9}    "
            f"{_format_accuracy(slices['rare'])}    "
            f"{_format_accuracy(slices['oov'])}"
        )


def _load_cached_datasets(
    *,
    cache_path: Path,
    expected_metadata: Mapping[str, object],
) -> tuple[MorphologyProbeDataset, dict[str, MorphologyProbeDataset]]:
    raw_cache = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw_cache, Mapping):
        raise ValueError("Morphology probe cache must contain a mapping.")
    if raw_cache.get("format_version") != MORPHOLOGY_PROBE_CACHE_FORMAT_VERSION:
        raise ValueError("Morphology probe cache format is incompatible.")
    if raw_cache.get("metadata") != dict(expected_metadata):
        raise ValueError(
            "Morphology probe cache does not match the checkpoint, schema, "
            "treebank release, or requested evaluation profiles."
        )
    raw_training = raw_cache.get("training")
    raw_evaluation = raw_cache.get("evaluation")
    if not isinstance(raw_training, Mapping) or not isinstance(raw_evaluation, Mapping):
        raise ValueError("Morphology probe cache datasets are invalid.")
    if not all(
        isinstance(language_tag, str) and isinstance(dataset, Mapping)
        for language_tag, dataset in raw_evaluation.items()
    ):
        raise ValueError("Morphology probe evaluation cache is invalid.")

    return (
        deserialize_morphology_probe_dataset(raw_training),
        {
            language_tag: deserialize_morphology_probe_dataset(dataset)
            for language_tag, dataset in raw_evaluation.items()
        },
    )


def main() -> None:
    arguments = parse_probe_arguments()
    checkpoint = torch.load(
        arguments.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Token-task checkpoint must contain a mapping.")
    validate_token_task_checkpoint_format(checkpoint)

    checkpoint_treebank_release = checkpoint.get("treebank_release", "current")
    if checkpoint_treebank_release != arguments.treebank_release:
        raise ValueError(
            "Checkpoint treebank release does not match the probe request: "
            f"{checkpoint_treebank_release!r}."
        )
    checkpoint_language_tag = checkpoint.get("language_tag")
    if not isinstance(checkpoint_language_tag, str):
        raise ValueError("Checkpoint language tag is invalid.")

    training_profiles = norwegian_training_profiles_for_language_tag(
        checkpoint_language_tag,
        treebank_release=arguments.treebank_release,
    )
    evaluation_language_tags = (
        arguments.evaluation_language_tags
        if arguments.evaluation_language_tags
        else tuple(profile.language_tag for profile in training_profiles)
    )
    if any(
        not norwegian_model_supports_language_tag(
            checkpoint_language_tag,
            language_tag,
        )
        for language_tag in evaluation_language_tags
    ):
        raise ValueError(
            "Checkpoint does not support every requested evaluation profile."
        )

    raw_schema_language_tags = checkpoint.get("schema_language_tags")
    if raw_schema_language_tags is None:
        schema_language_tags = tuple(
            profile.language_tag for profile in training_profiles
        )
    elif isinstance(raw_schema_language_tags, (list, tuple)) and all(
        isinstance(language_tag, str) for language_tag in raw_schema_language_tags
    ):
        schema_language_tags = tuple(raw_schema_language_tags)
    else:
        raise ValueError("Checkpoint schema language tags are invalid.")

    schema_training_tokens = tuple(
        sentence
        for language_tag in schema_language_tags
        for sentence in read_sentences(
            norwegian_profile_for_language_tag(
                language_tag,
                treebank_release=arguments.treebank_release,
            ).gold_treebank.training_path
        )
    )
    schema = build_norwegian_schema(schema_training_tokens)
    serialized_schema = serialize_token_task_schema(schema)
    if checkpoint.get("schema") != serialized_schema:
        raise ValueError("Checkpoint schema does not match the pinned training data.")

    training_tokens = tuple(
        sentence
        for profile in training_profiles
        for sentence in read_sentences(profile.gold_treebank.training_path)
    )
    development_tokens = {
        language_tag: read_sentences(
            norwegian_profile_for_language_tag(
                language_tag,
                treebank_release=arguments.treebank_release,
            ).gold_treebank.development_path
        )
        for language_tag in evaluation_language_tags
    }
    checkpoint_sha256 = _file_sha256(arguments.checkpoint_path)
    cache_metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "treebank_release": arguments.treebank_release,
        "schema": serialized_schema,
        "evaluation_language_tags": evaluation_language_tags,
        "representation_boundary": "morphology-pre-head",
    }

    training_dataset: MorphologyProbeDataset
    evaluation_datasets: dict[str, MorphologyProbeDataset]
    cache_path = arguments.representation_cache_path
    if (
        cache_path is not None
        and cache_path.exists()
        and not arguments.rebuild_representation_cache
    ):
        print("Loading representation cache:", cache_path, flush=True)
        training_dataset, evaluation_datasets = _load_cached_datasets(
            cache_path=cache_path,
            expected_metadata=cache_metadata,
        )
    else:
        raw_model_role = checkpoint.get("model_role", "student")
        if raw_model_role not in ("student", "teacher"):
            raise ValueError("Checkpoint model role is invalid.")
        model_role = cast(ModelRole, raw_model_role)
        backbone_spec = training_profiles[0].backbone_for_role(model_role)
        if checkpoint.get("backbone_model_id") != backbone_spec.model_id:
            raise ValueError("Checkpoint backbone model does not match.")
        if checkpoint.get("backbone_revision") != backbone_spec.revision:
            raise ValueError("Checkpoint backbone revision does not match.")

        head_architecture = token_task_head_architecture_from_checkpoint(checkpoint)
        character_vocabulary = character_vocabulary_from_checkpoint(
            checkpoint,
            architecture=head_architecture,
        )
        maximum_character_count = maximum_character_count_from_checkpoint(
            checkpoint,
            architecture=head_architecture,
        )
        model = build_pretrained_token_tagger(
            backbone_spec=backbone_spec,
            schema=schema,
            dropout_probability=0.1,
            pooling_strategy=token_pooling_strategy_from_checkpoint(checkpoint),
            head_architecture=head_architecture,
            morphology_pre_head_architecture=(
                morphology_pre_head_architecture_from_checkpoint(checkpoint)
            ),
            layer_aggregation_strategy=(
                backbone_layer_aggregation_strategy_from_checkpoint(checkpoint)
            ),
            character_vocabulary_size=(
                None if character_vocabulary is None else character_vocabulary.size
            ),
            morphology_bundle_reranker_spec=(
                morphology_bundle_reranker_spec_from_checkpoint(checkpoint)
            ),
            morphology_agreement_refiner_spec=(
                morphology_agreement_refiner_spec_from_checkpoint(checkpoint)
            ),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        device = torch.device(arguments.device)
        model.to(device)

        tokenizer = load_backbone_tokenizer(backbone_spec)
        extraction_batch_size = int(checkpoint["training_config"]["batch_size"])
        training_corpus = encode_norwegian_sentences(
            training_tokens,
            schema=schema,
        )
        training_sentence_batches = _sentence_batches(
            training_corpus.sentences,
            batch_size=extraction_batch_size,
        )
        print(
            "Extracting frozen training representations:",
            training_corpus.token_count,
            "tokens",
            flush=True,
        )
        training_dataset = extract_morphology_probe_dataset(
            model=model,
            batches=iter_supervised_token_task_batches(
                tokenizer=tokenizer,
                sentence_batches=training_sentence_batches,
                character_vocabulary=character_vocabulary,
                maximum_character_count=(
                    32 if maximum_character_count is None else maximum_character_count
                ),
            ),
            device=device,
            on_batch=_progress_callback(
                label="Training extraction",
                total=len(training_sentence_batches),
            ),
        )

        frequency_profile = TokenFrequencyProfile.from_token_sequences(
            tuple(
                tuple(token.text for token in sentence) for sentence in training_tokens
            )
        )
        evaluation_datasets = {}
        for language_tag, raw_sentences in development_tokens.items():
            development_corpus = encode_norwegian_sentences(
                raw_sentences,
                schema=schema,
            )
            sentence_batches = _sentence_batches(
                development_corpus.sentences,
                batch_size=extraction_batch_size,
            )
            pretokenized_batches = tuple(
                tuple(sentence.model_input for sentence in sentence_batch)
                for sentence_batch in sentence_batches
            )
            slice_masks = {
                frequency_class.value: frequency_profile.build_batch_masks(
                    pretokenized_batches,
                    frequency_class=frequency_class,
                )
                for frequency_class in (
                    TokenFrequencyClass.RARE,
                    TokenFrequencyClass.OOV,
                )
            }
            print(
                f"Extracting frozen {language_tag} representations:",
                development_corpus.token_count,
                "tokens",
                flush=True,
            )
            evaluation_datasets[language_tag] = extract_morphology_probe_dataset(
                model=model,
                batches=iter_supervised_token_task_batches(
                    tokenizer=tokenizer,
                    sentence_batches=sentence_batches,
                    character_vocabulary=character_vocabulary,
                    maximum_character_count=(
                        32
                        if maximum_character_count is None
                        else maximum_character_count
                    ),
                ),
                device=device,
                batch_slice_masks=slice_masks,
                on_batch=_progress_callback(
                    label=f"{language_tag} extraction",
                    total=len(sentence_batches),
                ),
            )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": MORPHOLOGY_PROBE_CACHE_FORMAT_VERSION,
                    "metadata": cache_metadata,
                    "training": serialize_morphology_probe_dataset(training_dataset),
                    "evaluation": {
                        language_tag: serialize_morphology_probe_dataset(dataset)
                        for language_tag, dataset in evaluation_datasets.items()
                    },
                },
                cache_path,
            )
            print("Representation cache:", cache_path, flush=True)

    morphology_weights = morphology_weights_from_checkpoint(checkpoint)
    device = torch.device(arguments.device)
    serialized_results: dict[str, object] = {}
    for architecture in arguments.architectures:
        print()
        print("Training frozen-head probe:", architecture.value, flush=True)
        probe, epoch_losses = train_morphology_head_probe(
            dataset=training_dataset,
            morphology_schema=schema.morphology,
            architecture=architecture,
            config=arguments.config,
            device=device,
            morphology_weights=morphology_weights,
            on_epoch=lambda epoch, loss, name=architecture.value: print(
                f"{name}: epoch {epoch}/{arguments.config.epoch_count} loss={loss:.6f}",
                flush=True,
            ),
        )
        parameter_count = sum(parameter.numel() for parameter in probe.parameters())
        print("Probe parameters:", parameter_count)

        evaluation_results: dict[str, object] = {}
        for language_tag, dataset in evaluation_datasets.items():
            metrics = evaluate_morphology_head_probe(
                probe=probe,
                dataset=dataset,
                morphology_schema=schema.morphology,
                device=device,
                batch_size=arguments.config.batch_size,
                morphology_weights=morphology_weights,
                logit_correction_strength=(
                    arguments.morphology_logit_correction_strength
                ),
            )
            _print_feature_metrics(
                language_tag=language_tag,
                architecture=architecture,
                metrics=metrics,
            )
            evaluation_results[language_tag] = {
                "token_count": dataset.token_count,
                "features": tuple(
                    _serialize_feature_metrics(feature) for feature in metrics
                ),
            }

        serialized_results[architecture.value] = {
            "parameter_count": parameter_count,
            "epoch_losses": epoch_losses,
            "evaluation": evaluation_results,
        }
        del probe

    arguments.analysis_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.analysis_path.write_text(
        json.dumps(
            {
                "format_version": MORPHOLOGY_PROBE_REPORT_FORMAT_VERSION,
                "diagnostic_only": True,
                "representation_boundary": "morphology-pre-head",
                "source_checkpoint": str(arguments.checkpoint_path),
                "source_checkpoint_sha256": checkpoint_sha256,
                "source_checkpoint_epoch": int(checkpoint["epoch_index"]) + 1,
                "treebank_release": arguments.treebank_release,
                "evaluation_language_tags": evaluation_language_tags,
                "training_token_count": training_dataset.token_count,
                "morphology_logit_correction_strength": (
                    arguments.morphology_logit_correction_strength
                ),
                "config": asdict(arguments.config),
                "results": serialized_results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print("Probe analysis:", arguments.analysis_path)
    print("Production checkpoint unchanged:", arguments.checkpoint_path)


if __name__ == "__main__":
    main()
