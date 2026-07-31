import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from prism.conllu import read_sentences
from prism.data import (
    NorwegianUdMorphologyDecoder,
    build_norwegian_schema,
    build_norwegian_ud_lemma_decoder,
    encode_norwegian_sentence,
    encode_norwegian_sentences,
)
from prism.evaluation.classification import (
    calculate_classification_metrics,
)
from prism.evaluation import (
    MorphologyErrorAuditAccumulator,
    MorphologyFeatureComparisonAccumulator,
    TokenFrequencyClass,
    TokenFrequencyProfile,
    UniversalDependenciesEvaluationAccumulator,
    UniversalFeaturesPolicyStep,
    build_universal_dependencies_reference_batch,
    serialize_morphology_feature_comparison_report,
    serialize_universal_dependencies_evaluation_metrics,
)
from prism.evaluation.reporting import (
    format_classification_metric_rows,
    format_morphology_error_attribution_rows,
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
    MorphologyBundleLossPolicy,
    TokenTaskLossWeights,
    audit_token_task_interactions,
    backbone_layer_aggregation_strategy_from_checkpoint,
    evenly_spaced_batch_indices,
    evaluate_supervised_token_task_epoch,
    iter_supervised_token_task_batches,
    token_pooling_strategy_from_checkpoint,
    token_task_head_architecture_from_checkpoint,
    validate_token_task_checkpoint_format,
    character_vocabulary_from_checkpoint,
    maximum_character_count_from_checkpoint,
    morphology_logit_correction_from_checkpoint,
    morphology_weights_from_checkpoint,
    morphology_pre_head_architecture_from_checkpoint,
    morphology_bundle_reranker_spec_from_checkpoint,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineEvaluationArguments:
    checkpoint_path: Path
    analysis_path: Path
    language_tag: str
    device: str
    treebank_release: str
    morphology_logit_correction_strength: float
    ud_morphology_policy: str
    disable_morphology_bundle_reranker: bool
    morphology_error_audit_feature: str | None
    morphology_error_audit_comparison_path: Path | None
    morphology_feature_comparison_path: Path | None
    morphology_feature_comparison_name: str
    task_interaction_audit: bool
    task_interaction_gradient_batch_count: int


def _norwegian_ud_morphology_policy_steps(
    *,
    language_tag: str,
) -> tuple[UniversalFeaturesPolicyStep, ...]:
    decoder = NorwegianUdMorphologyDecoder(language_tag=language_tag)
    steps = [
        UniversalFeaturesPolicyStep(
            name="common-gender",
            decoder=decoder.decode_common_gender,
        )
    ]
    if language_tag == "nn":
        steps.extend(
            (
                UniversalFeaturesPolicyStep(
                    name="nynorsk-number",
                    decoder=decoder.decode_nynorsk_number,
                ),
                UniversalFeaturesPolicyStep(
                    name="nynorsk-definite",
                    decoder=decoder.decode_nynorsk_definite,
                ),
            )
        )
    return tuple(steps)


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
    parser.add_argument(
        "--ud-morphology-policy",
        choices=("canonical", "treebank"),
        default="canonical",
        help=(
            "Map canonical morphology to the selected UD treebank's annotation "
            "convention before UFeats scoring."
        ),
    )
    parser.add_argument(
        "--disable-morphology-bundle-reranker",
        action="store_true",
        help="Disable a checkpointed bundle reranker for a matched ablation.",
    )
    parser.add_argument(
        "--morphology-error-audit-feature",
        help=(
            "Collect token-aligned errors for this morphology feature in the "
            "analysis JSON."
        ),
    )
    parser.add_argument(
        "--morphology-error-audit-comparison",
        type=Path,
        dest="morphology_error_audit_comparison_path",
        help=(
            "Optional aligned CoNLL-U prediction used to count which audited "
            "errors the comparison system solves."
        ),
    )
    parser.add_argument(
        "--morphology-feature-comparison",
        type=Path,
        dest="morphology_feature_comparison_path",
        help=(
            "Optional aligned CoNLL-U prediction used for a complete "
            "feature-by-feature morphology comparison. Omit this option for "
            "the faster standalone Prism evaluation; Prism never launches the "
            "comparison system itself."
        ),
    )
    parser.add_argument(
        "--morphology-feature-comparison-name",
        default="UDPipe",
        help="Display name for the aligned feature-comparison system.",
    )
    parser.add_argument(
        "--task-interaction-audit",
        action="store_true",
        help=(
            "Audit bundle and lemma ranks plus supervised task-gradient "
            "alignment without updating model parameters."
        ),
    )
    parser.add_argument(
        "--task-interaction-gradient-batches",
        type=int,
        default=16,
        dest="task_interaction_gradient_batch_count",
        help=(
            "Number of evenly spaced Development batches used for gradient "
            "alignment when --task-interaction-audit is enabled."
        ),
    )

    parsed_arguments = parser.parse_args(arguments)
    if not 0.0 <= parsed_arguments.morphology_logit_correction_strength <= 1.0:
        parser.error("--morphology-logit-correction-strength must be between 0 and 1")
    if (
        parsed_arguments.morphology_error_audit_comparison_path is not None
        and parsed_arguments.morphology_error_audit_feature is None
    ):
        parser.error(
            "--morphology-error-audit-comparison requires "
            "--morphology-error-audit-feature"
        )
    if (
        not parsed_arguments.morphology_feature_comparison_name
        or parsed_arguments.morphology_feature_comparison_name.strip()
        != parsed_arguments.morphology_feature_comparison_name
    ):
        parser.error(
            "--morphology-feature-comparison-name must be non-empty and trimmed"
        )
    if parsed_arguments.task_interaction_gradient_batch_count <= 0:
        parser.error("--task-interaction-gradient-batches must be greater than zero")
    if (
        parsed_arguments.task_interaction_audit
        and parsed_arguments.disable_morphology_bundle_reranker
    ):
        parser.error(
            "--task-interaction-audit cannot be combined with "
            "--disable-morphology-bundle-reranker"
        )

    return BaselineEvaluationArguments(
        language_tag=parsed_arguments.language_tag,
        checkpoint_path=parsed_arguments.checkpoint_path,
        analysis_path=parsed_arguments.analysis_path,
        device=parsed_arguments.device,
        treebank_release=parsed_arguments.treebank_release,
        morphology_logit_correction_strength=(
            parsed_arguments.morphology_logit_correction_strength
        ),
        ud_morphology_policy=parsed_arguments.ud_morphology_policy,
        disable_morphology_bundle_reranker=(
            parsed_arguments.disable_morphology_bundle_reranker
        ),
        morphology_error_audit_feature=(
            parsed_arguments.morphology_error_audit_feature
        ),
        morphology_error_audit_comparison_path=(
            parsed_arguments.morphology_error_audit_comparison_path
        ),
        morphology_feature_comparison_path=(
            parsed_arguments.morphology_feature_comparison_path
        ),
        morphology_feature_comparison_name=(
            parsed_arguments.morphology_feature_comparison_name
        ),
        task_interaction_audit=parsed_arguments.task_interaction_audit,
        task_interaction_gradient_batch_count=(
            parsed_arguments.task_interaction_gradient_batch_count
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

    backbone_spec = profile.backbone_for_model_id(
        checkpoint["backbone_model_id"],
        role=model_role,
    )
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
    universal_features_policy_steps = (
        ()
        if arguments.ud_morphology_policy == "canonical"
        else _norwegian_ud_morphology_policy_steps(language_tag=profile.language_tag)
    )
    morphology_error_audit_accumulator = None
    if arguments.morphology_error_audit_feature is not None:
        comparison_reference_batches = None
        if arguments.morphology_error_audit_comparison_path is not None:
            comparison_tokens = read_sentences(
                arguments.morphology_error_audit_comparison_path
            )
            comparison_reference_batches = tuple(
                build_universal_dependencies_reference_batch(
                    comparison_tokens[start : start + batch_size]
                )
                for start in range(0, len(comparison_tokens), batch_size)
            )
        morphology_error_audit_accumulator = MorphologyErrorAuditAccumulator(
            schema=schema,
            feature_name=arguments.morphology_error_audit_feature,
            reference_batches=development_reference_batches,
            frequency_profile=frequency_profile,
            comparison_reference_batches=comparison_reference_batches,
        )
    morphology_feature_comparison_accumulator = None
    if arguments.morphology_feature_comparison_path is not None:
        feature_comparison_tokens = read_sentences(
            arguments.morphology_feature_comparison_path
        )
        feature_comparison_reference_batches = tuple(
            build_universal_dependencies_reference_batch(
                feature_comparison_tokens[start : start + batch_size]
            )
            for start in range(0, len(feature_comparison_tokens), batch_size)
        )
        morphology_feature_comparison_accumulator = (
            MorphologyFeatureComparisonAccumulator(
                schema=schema,
                reference_batches=development_reference_batches,
                comparison_reference_batches=(feature_comparison_reference_batches),
                frequency_profile=frequency_profile,
                universal_features_policy_steps=(universal_features_policy_steps),
            )
        )

    tokenizer = load_backbone_tokenizer(backbone_spec)
    pooling_strategy = token_pooling_strategy_from_checkpoint(checkpoint)
    head_architecture = token_task_head_architecture_from_checkpoint(checkpoint)
    morphology_pre_head_architecture = morphology_pre_head_architecture_from_checkpoint(
        checkpoint
    )
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
    morphology_bundle_reranker_spec = (
        morphology_bundle_reranker_spec_from_checkpoint(checkpoint)
    )
    model = build_pretrained_token_tagger(
        backbone_spec=backbone_spec,
        schema=schema,
        dropout_probability=0.1,
        pooling_strategy=pooling_strategy,
        head_architecture=head_architecture,
        morphology_pre_head_architecture=morphology_pre_head_architecture,
        layer_aggregation_strategy=layer_aggregation_strategy,
        character_vocabulary_size=(
            None if character_vocabulary is None else character_vocabulary.size
        ),
        morphology_bundle_reranker_spec=morphology_bundle_reranker_spec,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    bundle_reranker = model.heads.morphology_bundle_reranker
    if arguments.disable_morphology_bundle_reranker:
        if bundle_reranker is None:
            raise ValueError(
                "Checkpoint does not contain a morphology bundle reranker."
            )
        bundle_reranker.set_enabled(False)
    print(
        "Evaluating checkpoint epoch:",
        int(checkpoint["epoch_index"]) + 1,
    )
    print("Treebank release:", arguments.treebank_release)
    print("Token pooling:", pooling_strategy.value)
    print("Task-head architecture:", head_architecture.value)
    print(
        "Morphology pre-head architecture:",
        morphology_pre_head_architecture.value,
    )
    print("Backbone layer aggregation:", layer_aggregation_strategy.value)
    print(
        "Morphology logit correction:",
        f"{arguments.morphology_logit_correction_strength:.2f}",
    )
    print("UD morphology policy:", arguments.ud_morphology_policy)
    print(
        "Morphology bundle reranker:",
        "disabled"
        if bundle_reranker is not None and not bundle_reranker.enabled
        else "enabled"
        if bundle_reranker is not None
        else "absent",
    )
    if bundle_reranker is not None:
        print(
            "Morphology bundle scorer:",
            bundle_reranker.spec.scorer_architecture.value,
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
                universal_features_policy_steps=universal_features_policy_steps,
            )
        ),
        prediction_observers=tuple(
            observer
            for observer in (
                morphology_error_audit_accumulator,
                morphology_feature_comparison_accumulator,
            )
            if observer is not None
        ),
        morphology_logit_correction=morphology_logit_correction,
    )

    if metrics.universal_dependencies is None:
        raise RuntimeError("UD-compatible metrics were not calculated.")
    morphology_error_audit = (
        None
        if morphology_error_audit_accumulator is None
        else morphology_error_audit_accumulator.finish()
    )
    morphology_feature_comparison = (
        None
        if morphology_feature_comparison_accumulator is None
        else morphology_feature_comparison_accumulator.finish()
    )
    task_interaction_audit = None
    if arguments.task_interaction_audit:
        if morphology_bundle_reranker_spec is None:
            raise ValueError(
                "Task interaction audit requires a checkpointed bundle reranker."
            )
        raw_morphology_weights = checkpoint.get("morphology_weights")
        audit_loss_weights = (
            None
            if raw_morphology_weights is None
            else TokenTaskLossWeights(
                morphology_weights=morphology_weights_from_checkpoint(checkpoint)
            )
        )
        candidate_targets = MorphologyBundleLossPolicy.from_reranker_spec(
            spec=morphology_bundle_reranker_spec,
            weight=0.0,
        ).candidate_morphology_targets
        gradient_batch_indices = evenly_spaced_batch_indices(
            batch_count=len(development_sentence_batches),
            selected_count=min(
                arguments.task_interaction_gradient_batch_count,
                len(development_sentence_batches),
            ),
        )
        task_interaction_audit = audit_token_task_interactions(
            model=model,
            batches=iter_supervised_token_task_batches(
                tokenizer=tokenizer,
                sentence_batches=development_sentence_batches,
                character_vocabulary=character_vocabulary,
                maximum_character_count=(
                    32
                    if maximum_character_count is None
                    else maximum_character_count
                ),
            ),
            device=torch.device(arguments.device),
            schema=schema,
            candidate_morphology_targets=candidate_targets,
            lemma_rule_training_counts=_lemma_rule_training_counts(
                schema_training_tokens=schema_training_tokens,
                schema=schema,
            ),
            token_slice_masks=token_slice_masks,
            gradient_batch_indices=gradient_batch_indices,
            loss_weights=audit_loss_weights,
            morphology_logit_correction=morphology_logit_correction,
        )

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

    if metrics.universal_dependencies.ufeats_policy_audits:
        print()
        print("UD morphology policy audit")
        for audit in metrics.universal_dependencies.ufeats_policy_audits:
            print(
                f"{audit.name:<20}  "
                f"changed={audit.changed_bundle_count:>5}  "
                f"improved={audit.improved_bundle_count:>5}  "
                f"regressed={audit.regressed_bundle_count:>5}"
            )

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
            universal_dependencies=token_slice.universal_dependencies,
        ):
            print(row)
        print()
        print(
            f"{token_slice.name.upper()} morphology feature-error attribution "
            "(share of all per-feature errors)"
        )
        for row in format_morphology_error_attribution_rows(
            slice_name=token_slice.name.upper(),
            feature_names=tuple(feature.name for feature in schema.morphology.features),
            metrics=token_slice.metrics,
        ):
            print(row)

    if morphology_error_audit is not None:
        print()
        print("Morphology error audit:", morphology_error_audit.feature_name)
        print(
            f"errors={morphology_error_audit.error_count} / "
            f"{morphology_error_audit.token_count} "
            f"({morphology_error_audit.error_count / morphology_error_audit.token_count:.4%})"
        )
        if morphology_error_audit.comparison_feature_correct_count is not None:
            print(
                "comparison feature-correct:",
                morphology_error_audit.comparison_feature_correct_count,
            )
            print(
                "comparison bundle-correct:",
                morphology_error_audit.comparison_bundle_correct_count,
            )

        print("By training frequency")
        for count in morphology_error_audit.frequency_class_counts:
            print(f"  {count.name:<12} {count.count:>5}")

        print("By gold UPOS")
        for count in morphology_error_audit.gold_upos_counts:
            print(f"  {count.name:<12} {count.count:>5}")

        print("Most frequent confusions")
        for confusion in morphology_error_audit.confusion_counts[:12]:
            gold = ",".join(confusion.gold_values) or "<NONE>"
            predicted = ",".join(confusion.predicted_values) or "<NONE>"
            print(f"  {gold:>10} -> {predicted:<10} {confusion.count:>5}")

        print("Most frequent gold-UPOS contexts")
        for context in morphology_error_audit.context_counts[:12]:
            label = f"{context.previous_upos}>{context.gold_upos}>{context.next_upos}"
            print(f"  {label:<32} {context.count:>5}")

        print("Most frequent normalized forms")
        for count in morphology_error_audit.normalized_form_counts[:20]:
            print(f"  {count.name:<24} {count.count:>5}")

    if morphology_feature_comparison is not None:
        print()
        print(
            "Morphology feature comparison:",
            arguments.morphology_feature_comparison_path,
        )
        feature_width = max(
            len(feature.feature_name)
            for feature in morphology_feature_comparison.features
        )
        comparison_name = arguments.morphology_feature_comparison_name
        overall_prism_header = "Prism overall"
        overall_comparison_header = f"{comparison_name} overall"
        annotated_prism_header = "Prism annotated"
        annotated_comparison_header = f"{comparison_name} annotated"
        delta_header = f"Prism-{comparison_name}"
        overall_prism_width = max(len(overall_prism_header), 8)
        overall_comparison_width = max(len(overall_comparison_header), 8)
        annotated_prism_width = max(len(annotated_prism_header), 9)
        annotated_comparison_width = max(len(annotated_comparison_header), 9)
        delta_width = max(len(delta_header), 9)
        print("Overall and annotated-token accuracy")
        print(
            f"{'Feature':<{feature_width}}    "
            f"{overall_prism_header:>{overall_prism_width}}    "
            f"{overall_comparison_header:>{overall_comparison_width}}    "
            f"{delta_header:>{delta_width}}    "
            f"{annotated_prism_header:>{annotated_prism_width}}    "
            f"{annotated_comparison_header:>{annotated_comparison_width}}    "
            f"{delta_header:>{delta_width}}"
        )
        for feature in morphology_feature_comparison.features:
            model_annotated = feature.model.annotated_accuracy
            comparison_annotated = feature.comparison.annotated_accuracy
            annotated_delta = (
                None
                if model_annotated is None or comparison_annotated is None
                else model_annotated - comparison_annotated
            )
            print(
                f"{feature.feature_name:<{feature_width}}    "
                f"{feature.model.overall_accuracy:>{overall_prism_width}.6f}    "
                f"{feature.comparison.overall_accuracy:>{overall_comparison_width}.6f}    "
                f"{feature.model.overall_accuracy - feature.comparison.overall_accuracy:>+{delta_width}.6f}    "
                f"{_format_optional_metric(model_annotated):>{annotated_prism_width}}    "
                f"{_format_optional_metric(comparison_annotated):>{annotated_comparison_width}}    "
                f"{_format_optional_delta(annotated_delta):>{delta_width}}"
            )

        print()
        print("Rare and OOV overall accuracy")
        rare_prism_header = "Prism rare"
        rare_comparison_header = f"{comparison_name} rare"
        oov_prism_header = "Prism OOV"
        oov_comparison_header = f"{comparison_name} OOV"
        rare_prism_width = max(len(rare_prism_header), 8)
        rare_comparison_width = max(len(rare_comparison_header), 8)
        oov_prism_width = max(len(oov_prism_header), 8)
        oov_comparison_width = max(len(oov_comparison_header), 8)
        print(
            f"{'Feature':<{feature_width}}    "
            f"{rare_prism_header:>{rare_prism_width}}    "
            f"{rare_comparison_header:>{rare_comparison_width}}    "
            f"{delta_header:>{delta_width}}    "
            f"{oov_prism_header:>{oov_prism_width}}    "
            f"{oov_comparison_header:>{oov_comparison_width}}    "
            f"{delta_header:>{delta_width}}"
        )
        for feature in morphology_feature_comparison.features:
            slices = {
                token_slice.name: token_slice
                for token_slice in feature.frequency_slices
            }
            rare = slices["rare"]
            oov = slices["oov"]
            print(
                f"{feature.feature_name:<{feature_width}}    "
                f"{rare.model.overall_accuracy:>{rare_prism_width}.6f}    "
                f"{rare.comparison.overall_accuracy:>{rare_comparison_width}.6f}    "
                f"{rare.model.overall_accuracy - rare.comparison.overall_accuracy:>+{delta_width}.6f}    "
                f"{oov.model.overall_accuracy:>{oov_prism_width}.6f}    "
                f"{oov.comparison.overall_accuracy:>{oov_comparison_width}.6f}    "
                f"{oov.model.overall_accuracy - oov.comparison.overall_accuracy:>+{delta_width}.6f}"
            )

    if task_interaction_audit is not None:
        print()
        print("Task interaction audit")
        bundle_audit = task_interaction_audit.morphology_bundles
        print(
            "Bundle coverage:",
            f"{bundle_audit.candidate_covered_count}/{bundle_audit.token_count}",
            f"({bundle_audit.candidate_coverage:.4%})",
        )
        print(
            "Final bundle errors:",
            bundle_audit.final_bundle_error_count,
        )
        print(
            "Error attribution:",
            f"uncovered={bundle_audit.uncovered_error_count}",
            f"ranking={bundle_audit.ranking_error_count}",
            f"refinement={bundle_audit.refinement_error_count}",
        )
        print(
            "Covered candidate ranks:",
            _format_rank_audit(bundle_audit.covered_ranks),
        )
        if bundle_audit.error_ranks is not None:
            print(
                "Error candidate ranks:",
                _format_rank_audit(bundle_audit.error_ranks),
            )
        print(
            "Gold margin:",
            f"overall={bundle_audit.mean_gold_margin:+.6f}",
            "errors="
            + (
                "undefined"
                if bundle_audit.mean_error_gold_margin is None
                else f"{bundle_audit.mean_error_gold_margin:+.6f}"
            ),
        )

        lemma_audit = task_interaction_audit.lemma_rules
        print("Lemma gold-rule ranks:", _format_rank_audit(lemma_audit.overall))
        print("Lemma ranks by token frequency")
        for group in lemma_audit.by_token_frequency:
            print(f"  {group.name:<12} {_format_rank_audit(group.metrics)}")
        print("Lemma ranks by rule frequency")
        for group in lemma_audit.by_rule_frequency:
            print(f"  {group.name:<12} {_format_rank_audit(group.metrics)}")

        print(
            "Gradient alignment:",
            task_interaction_audit.gradient_objective,
            f"batches={len(task_interaction_audit.gradient_batch_indices)}",
        )
        for group in task_interaction_audit.gradient_groups:
            print(
                f"  {group.name} ({group.parameter_count:,} parameters)"
            )
            for pair in group.task_pairs:
                print(
                    f"    {pair.first_task:<10} vs {pair.second_task:<10} "
                    f"cos={pair.mean_cosine_similarity:+.6f}  "
                    f"conflict={pair.conflict_rate:.2%}  "
                    f"n={pair.sample_count}"
                )

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
    serialized_metrics["token_slices"] = [
        {
            "name": token_slice.name,
            "metrics": asdict(token_slice.metrics),
            "universal_dependencies": (
                None
                if token_slice.universal_dependencies is None
                else serialize_universal_dependencies_evaluation_metrics(
                    token_slice.universal_dependencies
                )
            ),
        }
        for token_slice in metrics.token_slices
    ]
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
                    "ud_morphology_policy": arguments.ud_morphology_policy,
                    "morphology_bundle_reranker": (
                        "disabled"
                        if bundle_reranker is not None and not bundle_reranker.enabled
                        else "enabled"
                        if bundle_reranker is not None
                        else "absent"
                    ),
                    "task_interaction_audit": arguments.task_interaction_audit,
                },
                "schema": (serialize_token_task_schema(schema)),
                "metrics": serialized_metrics,
                "morphology_error_audit": (
                    None
                    if morphology_error_audit is None
                    else {
                        "comparison": (
                            None
                            if arguments.morphology_error_audit_comparison_path is None
                            else str(arguments.morphology_error_audit_comparison_path)
                        ),
                        "metrics": asdict(morphology_error_audit),
                    }
                ),
                "morphology_feature_comparison": (
                    None
                    if morphology_feature_comparison is None
                    else {
                        "comparison": str(arguments.morphology_feature_comparison_path),
                        "comparison_name": (
                            arguments.morphology_feature_comparison_name
                        ),
                        "ud_morphology_policy": arguments.ud_morphology_policy,
                        "metrics": (
                            serialize_morphology_feature_comparison_report(
                                morphology_feature_comparison
                            )
                        ),
                    }
                ),
                "task_interaction_audit": (
                    None
                    if task_interaction_audit is None
                    else asdict(task_interaction_audit)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Analysis:", analysis_path)


def _format_optional_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def _format_optional_delta(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.6f}"


def _format_rank_audit(metrics: object) -> str:
    from prism.training import RankAuditMetrics

    if not isinstance(metrics, RankAuditMetrics):
        raise TypeError("Rank formatter requires RankAuditMetrics.")
    return (
        f"n={metrics.count}  "
        f"top1={metrics.top1_accuracy:.4%}  "
        f"top2={metrics.top2_accuracy:.4%}  "
        f"top5={metrics.top5_accuracy:.4%}  "
        f"mean={metrics.mean_rank:.3f}  "
        f"mrr={metrics.mean_reciprocal_rank:.6f}"
    )


def _lemma_rule_training_counts(
    *,
    schema_training_tokens: Sequence[Sequence[object]],
    schema: object,
) -> tuple[int, ...]:
    from prism.conllu import Token
    from prism.schema import TokenTaskSchema

    if not isinstance(schema, TokenTaskSchema):
        raise TypeError("Lemma-rule counting requires TokenTaskSchema.")
    counts = [0] * len(schema.lemma_rules.rules)
    for raw_sentence in schema_training_tokens:
        if not all(isinstance(token, Token) for token in raw_sentence):
            raise TypeError("Lemma-rule counting requires CoNLL-U tokens.")
        sentence = encode_norwegian_sentence(
            cast(Sequence[Token], raw_sentence),
            schema=schema,
        )
        for target in sentence.targets:
            if target.lemma_rule_id is not None:
                counts[target.lemma_rule_id] += 1
    if any(count <= 0 for count in counts):
        raise RuntimeError("Every schema lemma rule must occur in training.")
    return tuple(counts)


if __name__ == "__main__":
    main()
