"""Export the selected Norwegian checkpoint as a versioned model artifact.

The resulting directory is the cross-platform release contract from
``docs/MODEL_STRATEGY.md``: a lowered ExecuTorch program with static shapes,
the manifest, the label schema, the subword vocabulary, recorded parity
fixtures, and license provenance. The morphology logit correction is embedded
in the exported graph, so native runtimes decode plain logits.

The artifact records canonical decoding only. The versioned Nynorsk external
treebank policy and confidence calibration remain documented follow-up tasks
before a production release.
"""

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from prism.conllu import read_sentences
from prism.data import encode_norwegian_sentences
from prism.data.examples import PretokenizedSentence
from prism.exporting import (
    CalibratedProbabilityExportLayer,
    CharacterAwareTokenTaggerExportAdapter,
    CheckpointProvenance,
    BackboneProvenance,
    FixedExportShapes,
    ModelArtifactManifest,
    ModelDataFileEntry,
    ModelProgramEntry,
    ParityFixtureBatch,
    TensorSpec,
    TokenTaggerExportAdapter,
    TokenizerContract,
    TreebankProvenance,
    build_fixtures_payload,
    build_labels_payload,
    decoded_sentence_predictions,
    fold_scaled_linear_parametrizations,
    lower_to_executorch_xnnpack,
    quantize_adapter_int8,
    maximum_task_probability_difference,
    pad_character_token_batch,
    pad_tokenized_batch,
    repeat_pad_sentences,
    run_executorch_program,
    sha256_of_bytes,
    sha256_of_file,
    token_task_logits_from_flat_outputs,
    top_k_output_tensors,
    write_json_file,
)
from prism.languages.norwegian import (
    norwegian_training_profiles_for_language_tag,
)
from prism.languages.norwegian.checkpoint_loading import (
    LoadedNorwegianTagger,
    load_norwegian_token_tagger,
)
from prism.modeling import (
    encode_character_token_batch,
    tokenize_pretokenized_sentences,
)
from prism.schema.characters import CHARACTER_PADDING_ID
from prism.training import morphology_logit_correction_from_checkpoint
from prism.training.calibration import (
    TaskTemperatureCalibration,
    load_task_temperature_calibration,
)


ARTIFACT_TASKS = ("upos", "morphology", "lemma_rules")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactExportArguments:
    checkpoint_path: Path
    artifact_version: str
    output_root: Path
    language_tag: str
    treebank_release: str
    morphology_logit_correction_strength: float
    batch_size: int
    subword_count: int
    token_count: int
    fixture_sentence_count: int
    fixture_lemma_top_k: int
    parity_tolerance: float
    overwrite: bool
    calibration_path: Path | None
    precision: str
    small_shapes: tuple[tuple[int, int], ...] | None
    external_data: bool


def parse_artifact_export_arguments(
    arguments: Sequence[str] | None = None,
) -> ArtifactExportArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Norwegian checkpoint as a versioned ExecuTorch artifact."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        dest="checkpoint_path",
    )
    parser.add_argument(
        "--artifact-version",
        required=True,
        help="Semantic artifact version, for example 0.1.0.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models"),
        help="Directory receiving the versioned artifact directory.",
    )
    parser.add_argument(
        "--language-tag",
        choices=("nb", "nn", "no"),
        default="no",
        help="Language coverage the exported artifact claims.",
    )
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parser.add_argument(
        "--morphology-logit-correction-strength",
        type=float,
        default=0.25,
        help=(
            "Correction embedded into the exported graph; 0.25 is the "
            "selected canonical strength of the shipped Norwegian student."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Fixed sentence count per exported inference batch.",
    )
    parser.add_argument(
        "--subword-count",
        type=int,
        default=160,
        help="Fixed subword length of the exported graph.",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=96,
        help="Fixed token length of the exported graph.",
    )
    parser.add_argument(
        "--fixture-sentence-count",
        type=int,
        default=8,
        help="Development sentences recorded per written standard.",
    )
    parser.add_argument(
        "--fixture-lemma-top-k",
        type=int,
        default=8,
        help=(
            "Record only the top-k lemma-rule logits per token; the "
            "complete lemma head spans thousands of edit rules and would "
            "dominate the fixture file."
        ),
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=5e-3,
        help=(
            "Largest allowed task-probability difference between eager "
            "PyTorch and the lowered ExecuTorch program at valid token "
            "positions; decoded predictions must match exactly."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        dest="calibration_path",
        help=(
            "Per-head temperature calibration JSON; when given, temperature "
            "scaling and softmax/sigmoid are baked into the graph and the "
            "artifact emits calibrated probabilities instead of logits."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "fp16-backbone", "int8"),
        default="fp32",
        help=(
            "Precision of the lowered program weights. fp16-backbone halves "
            "the backbone while task heads, correction, and calibration stay "
            "fp32. int8 quantizes linears dynamically and embeddings "
            "per-channel; fixtures then record the quantized eager twin and "
            "the runtime parity gate moves to the C++ test suite (the "
            "Python runtime lacks the quantized kernels)."
        ),
    )
    parser.add_argument(
        "--small-shapes",
        type=int,
        nargs=2,
        action="append",
        metavar=("SUBWORDS", "TOKENS"),
        default=None,
        help=(
            "Additionally lower a program with these smaller fixed shapes; "
            "repeat for several programs. Runtimes pick the smallest "
            "program a batch fits into."
        ),
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help=(
            "Store the model weights once in a shared model.ptd file instead "
            "of duplicating them inside every fixed-shape program; runtimes "
            "load the data file alongside each program."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact directory of the same version.",
    )
    parsed = parser.parse_args(arguments)
    return ArtifactExportArguments(
        checkpoint_path=parsed.checkpoint_path,
        artifact_version=parsed.artifact_version,
        output_root=parsed.output_root,
        language_tag=parsed.language_tag,
        treebank_release=parsed.treebank_release,
        morphology_logit_correction_strength=(
            parsed.morphology_logit_correction_strength
        ),
        batch_size=parsed.batch_size,
        subword_count=parsed.subword_count,
        token_count=parsed.token_count,
        fixture_sentence_count=parsed.fixture_sentence_count,
        fixture_lemma_top_k=parsed.fixture_lemma_top_k,
        parity_tolerance=parsed.parity_tolerance,
        overwrite=parsed.overwrite,
        calibration_path=parsed.calibration_path,
        precision=parsed.precision,
        small_shapes=(
            None
            if parsed.small_shapes is None
            else tuple(
                (subword_count, token_count)
                for subword_count, token_count in parsed.small_shapes
            )
        ),
        external_data=parsed.external_data,
    )


def _pretokenized_development_sentences(
    development_path: Path,
    tagger: LoadedNorwegianTagger,
) -> tuple[PretokenizedSentence, ...]:
    corpus = encode_norwegian_sentences(
        tuple(read_sentences(development_path)),
        schema=tagger.schema,
    )
    return tuple(sentence.model_input for sentence in corpus.sentences)


def _fitting_fixture_sentences(
    *,
    sentences: tuple[PretokenizedSentence, ...],
    tokenizer: PreTrainedTokenizerBase,
    shapes: FixedExportShapes,
    fixture_sentence_count: int,
) -> tuple[PretokenizedSentence, ...]:
    selected: list[PretokenizedSentence] = []
    for sentence in sentences:
        if len(sentence.tokens) > shapes.token_count:
            continue
        tokenized = tokenize_pretokenized_sentences(
            tokenizer=tokenizer,
            sentences=(sentence,),
        )
        if tokenized.max_subword_count > shapes.subword_count:
            continue
        selected.append(sentence)
        if len(selected) == fixture_sentence_count:
            break

    if len(selected) < fixture_sentence_count:
        raise ValueError(
            f"Development split provides only {len(selected)} of "
            f"{fixture_sentence_count} fixture sentences fitting the fixed "
            "export shapes."
        )
    return tuple(selected)


def _padding_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    padding_token_id = tokenizer.pad_token_id
    if not isinstance(padding_token_id, int) or padding_token_id < 0:
        raise ValueError("Export requires a tokenizer with a padding token ID.")
    return padding_token_id


def build_fixture_input_tensors(
    *,
    sentences: tuple[PretokenizedSentence, ...],
    tagger: LoadedNorwegianTagger,
    shapes: FixedExportShapes,
) -> tuple[tuple[str, Tensor], ...]:
    """Tokenize, encode, and pad one sentence batch to the export contract."""

    batch_sentences = repeat_pad_sentences(sentences, batch_size=shapes.batch_size)
    tokenized = pad_tokenized_batch(
        tokenize_pretokenized_sentences(
            tokenizer=tagger.tokenizer,
            sentences=batch_sentences,
        ),
        shapes=shapes,
        padding_token_id=_padding_token_id(tagger.tokenizer),
    )
    inputs: list[tuple[str, Tensor]] = [
        ("input_ids", tokenized.input_ids),
        ("attention_mask", tokenized.attention_mask),
        ("first_subword_indices", tokenized.first_subword_indices),
        ("subword_end_indices", tokenized.subword_end_indices),
        ("token_mask", tokenized.token_mask),
    ]

    if tagger.character_vocabulary is not None:
        characters = pad_character_token_batch(
            encode_character_token_batch(
                token_sequences=tuple(sentence.tokens for sentence in batch_sentences),
                vocabulary=tagger.character_vocabulary,
                maximum_character_count=tagger.maximum_character_count,
            ),
            shapes=shapes,
            character_padding_id=CHARACTER_PADDING_ID,
        )
        inputs.append(("character_ids", characters.character_ids))
        inputs.append(("character_mask", characters.character_mask))

    return tuple(inputs)


def _output_names(
    tagger: LoadedNorwegianTagger,
    *,
    calibrated: bool = False,
) -> tuple[str, ...]:
    kind = "probabilities" if calibrated else "logits"
    return (
        f"upos_{kind}",
        *(
            f"morphology_{kind}:{feature.name}"
            for feature in tagger.schema.morphology.features
        ),
        f"lemma_rule_{kind}",
    )


def _decodable_flat_outputs(
    outputs: tuple[Tensor, ...],
    *,
    tagger: LoadedNorwegianTagger,
    calibrated: bool,
) -> tuple[Tensor, ...]:
    """Map calibrated probabilities back to decode-equivalent logits.

    ``log`` preserves the argmax of exclusive heads exactly and ``logit``
    maps the multi-valued 0.5 probability threshold back to the 0.0 logit
    threshold, so the existing decoding and parity machinery stays exact.
    """

    if not calibrated:
        return outputs
    epsilon = 1e-7
    features = tagger.schema.morphology.features
    transformed = [torch.log(outputs[0].float().clamp_min(epsilon))]
    for feature, feature_outputs in zip(features, outputs[1 : 1 + len(features)]):
        probabilities = feature_outputs.float().clamp(epsilon, 1.0 - epsilon)
        if feature.allows_multiple_values:
            transformed.append(torch.logit(probabilities))
        else:
            transformed.append(torch.log(probabilities))
    transformed.append(torch.log(outputs[-1].float().clamp_min(epsilon)))
    return tuple(transformed)


def _tensor_specs(
    input_tensors: tuple[tuple[str, Tensor], ...],
) -> tuple[TensorSpec, ...]:
    dtype_names = {torch.int64: "int64", torch.float32: "float32", torch.bool: "bool"}
    return tuple(
        TensorSpec(
            name=name,
            dtype=dtype_names[tensor.dtype],
            shape=tuple(tensor.shape),
        )
        for name, tensor in input_tensors
    )


def build_export_adapter(
    tagger: LoadedNorwegianTagger,
    *,
    morphology_logit_correction_strength: float,
    calibration: "TaskTemperatureCalibration | None" = None,
) -> nn.Module:
    correction = morphology_logit_correction_from_checkpoint(
        tagger.checkpoint,
        strength=morphology_logit_correction_strength,
    )
    calibrated_probabilities = None
    if calibration is not None:
        if (
            calibration.morphology_logit_correction_strength
            != morphology_logit_correction_strength
        ):
            raise ValueError(
                "Calibration was fitted for a different logit-correction "
                f"strength: {calibration.morphology_logit_correction_strength}."
            )
        feature_names = tuple(
            feature.name for feature in tagger.schema.morphology.features
        )
        if feature_names != calibration.morphology_feature_names:
            raise ValueError("Calibration morphology features do not match the schema.")
        calibrated_probabilities = CalibratedProbabilityExportLayer(
            upos_temperature=calibration.upos_temperature,
            morphology_temperatures=calibration.morphology_temperatures,
            multi_valued_features=tuple(
                feature.allows_multiple_values
                for feature in tagger.schema.morphology.features
            ),
            lemma_rule_temperature=calibration.lemma_rule_temperature,
        )
    tagger.model.eval()
    if tagger.character_vocabulary is not None:
        return CharacterAwareTokenTaggerExportAdapter(
            model=tagger.model,
            morphology_logit_correction=correction,
            calibrated_probabilities=calibrated_probabilities,
        )
    return TokenTaggerExportAdapter(
        model=tagger.model,
        morphology_logit_correction=correction,
        calibrated_probabilities=calibrated_probabilities,
    )


def write_vocabulary_file(
    *,
    tokenizer: PreTrainedTokenizerBase,
    artifact_directory: Path,
    file_name: str = "vocabulary.json",
) -> TokenizerContract:
    """Save the backbone tokenizer definition into the artifact directory."""

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        tokenizer.save_pretrained(staging)
        tokenizer_file = staging / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise ValueError(
                "Export requires a fast tokenizer that serializes to tokenizer.json."
            )
        shutil.copyfile(tokenizer_file, artifact_directory / file_name)

    return TokenizerContract(
        file_name=file_name,
        class_name=type(tokenizer).__name__,
        padding_token_id=_padding_token_id(tokenizer),
    )


def write_licenses_directory(
    *,
    artifact_directory: Path,
    backbone: BackboneProvenance,
    treebanks: tuple[TreebankProvenance, ...],
    silver_corpus_paths: tuple[str, ...],
) -> None:
    licenses_directory = artifact_directory / "LICENSES"
    licenses_directory.mkdir()

    lines = [
        "# Model artifact provenance and licenses",
        "",
        "The exported model weights are derived from the following sources.",
        "",
        "## Backbone",
        "",
        f"- model: `{backbone.model_id}`",
        f"- revision: `{backbone.revision}`",
        "- license: see the model card of the backbone repository",
        "",
        "## Gold treebanks",
        "",
    ]
    for treebank in treebanks:
        lines.append(
            f"- `{treebank.repository_id}` at `{treebank.revision}` "
            f"({treebank.license_id})"
        )
    lines.extend(
        [
            "",
            "## Silver corpora",
            "",
        ]
    )
    if silver_corpus_paths:
        lines.extend(
            f"- `{corpus_path}` (provenance documented in `docs/PROJECT_STATUS.md`)"
            for corpus_path in silver_corpus_paths
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Prism source code is licensed under the Apache License 2.0; the",
            "model weights and datasets retain their own licenses and",
            "attribution requirements.",
            "",
        ]
    )
    (licenses_directory / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _silver_corpus_paths(checkpoint: dict) -> tuple[str, ...]:
    silver_training = checkpoint.get("silver_training")
    if not isinstance(silver_training, dict):
        return ()
    corpus_paths = silver_training.get("corpus_paths")
    if not isinstance(corpus_paths, (list, tuple)):
        return ()
    return tuple(str(corpus_path) for corpus_path in corpus_paths)


def main() -> None:
    arguments = parse_artifact_export_arguments()

    profiles = norwegian_training_profiles_for_language_tag(
        arguments.language_tag,
        treebank_release=arguments.treebank_release,
    )
    tagger = load_norwegian_token_tagger(
        checkpoint_path=arguments.checkpoint_path,
        required_language_tags=tuple(profile.language_tag for profile in profiles),
        treebank_release=arguments.treebank_release,
    )
    shapes = FixedExportShapes(
        batch_size=arguments.batch_size,
        subword_count=arguments.subword_count,
        token_count=arguments.token_count,
        character_count=(
            None
            if tagger.character_vocabulary is None
            else tagger.maximum_character_count
        ),
    )
    calibration = (
        None
        if arguments.calibration_path is None
        else load_task_temperature_calibration(arguments.calibration_path)
    )
    adapter = build_export_adapter(
        tagger,
        morphology_logit_correction_strength=(
            arguments.morphology_logit_correction_strength
        ),
        calibration=calibration,
    )
    lowering_adapter = adapter
    if arguments.precision == "fp16":
        adapter = adapter.to(torch.float16)
        lowering_adapter = adapter
        print("Precision: fp16 (weights and activations)")
    elif arguments.precision == "fp16-backbone":
        tagger.model.backbone.to(torch.float16)
        print("Precision: fp16 backbone, fp32 task heads")
    elif arguments.precision == "int8":
        folded = fold_scaled_linear_parametrizations(adapter)
        print(f"Folded {folded} scale-parametrized linears (exact).")
        calibration_batches = []
        for profile in profiles:
            calibration_sentences = _fitting_fixture_sentences(
                sentences=_pretokenized_development_sentences(
                    profile.gold_treebank.development_path,
                    tagger,
                ),
                tokenizer=tagger.tokenizer,
                shapes=shapes,
                fixture_sentence_count=arguments.fixture_sentence_count,
            )
            calibration_batches.append(
                tuple(
                    tensor
                    for _, tensor in build_fixture_input_tensors(
                        sentences=calibration_sentences,
                        tagger=tagger,
                        shapes=shapes,
                    )
                )
            )
        # Fixtures and quality references use the eager twin at the main
        # shapes; lowering re-quantizes the floating adapter per shape
        # (converted twins carry shape guards), which is deterministic for
        # weights, so all programs share one data file.
        lowering_adapter = adapter
        adapter = quantize_adapter_int8(
            adapter=adapter,
            calibration_batches=calibration_batches,
        )
        print(
            "Precision: int8 (dynamic linears, per-channel embeddings); "
            "fixtures record the quantized eager twin."
        )

    artifact_name = f"prism-{arguments.language_tag}"
    artifact_directory = (
        arguments.output_root / f"{artifact_name}-{arguments.artifact_version}"
    )
    if artifact_directory.exists():
        if not arguments.overwrite:
            raise SystemExit(
                f"Artifact directory {artifact_directory} already exists; "
                "pass --overwrite to replace it."
            )
        shutil.rmtree(artifact_directory)
    artifact_directory.mkdir(parents=True)

    print("Exporting checkpoint epoch:", tagger.epoch_index + 1)
    print("Language tags:", ", ".join(p.language_tag for p in profiles))
    print(
        "Morphology logit correction:",
        f"{arguments.morphology_logit_correction_strength:.2f}",
    )
    print(
        "Fixed shapes:",
        f"batch={shapes.batch_size}",
        f"subwords={shapes.subword_count}",
        f"tokens={shapes.token_count}",
        f"characters={shapes.character_count}",
    )

    fixtures: list[ParityFixtureBatch] = []
    eager_outputs_by_fixture: list[tuple[Tensor, ...]] = []
    for profile in profiles:
        sentences = _fitting_fixture_sentences(
            sentences=_pretokenized_development_sentences(
                profile.gold_treebank.development_path,
                tagger,
            ),
            tokenizer=tagger.tokenizer,
            shapes=shapes,
            fixture_sentence_count=arguments.fixture_sentence_count,
        )
        batch_sentences = repeat_pad_sentences(
            sentences,
            batch_size=shapes.batch_size,
        )
        input_tensors = build_fixture_input_tensors(
            sentences=sentences,
            tagger=tagger,
            shapes=shapes,
        )
        with torch.no_grad():
            eager_outputs = tuple(
                output.detach().clone()
                for output in adapter(*(tensor for _, tensor in input_tensors))
            )
        decoded = decoded_sentence_predictions(
            schema=tagger.schema,
            logits=token_task_logits_from_flat_outputs(
                _decodable_flat_outputs(
                    eager_outputs,
                    tagger=tagger,
                    calibrated=calibration is not None,
                ),
                schema=tagger.schema,
            ),
            token_mask=dict(input_tensors)["token_mask"],
            sentence_tokens=tuple(sentence.tokens for sentence in batch_sentences),
        )
        eager_outputs_by_fixture.append(eager_outputs)
        output_names = _output_names(tagger, calibrated=calibration is not None)
        fixtures.append(
            ParityFixtureBatch(
                name=f"{profile.language_tag}-development",
                input_tensors=input_tensors,
                expected_output_tensors=(
                    *zip(output_names[:-1], eager_outputs[:-1], strict=True),
                    *top_k_output_tensors(
                        name=output_names[-1],
                        logits=eager_outputs[-1],
                        top_k=arguments.fixture_lemma_top_k,
                    ),
                ),
                decoded_sentences=tuple(decoded),
            )
        )
        print(
            f"Recorded fixture batch {profile.language_tag}-development:",
            f"{len(sentences)} sentences",
        )

    example_inputs = tuple(tensor for _, tensor in fixtures[0].input_tensors)
    external_data_name = "model" if arguments.external_data else None
    data_path: Path | None = None
    program_data_files: tuple[str, ...] = ()
    print("Lowering to ExecuTorch XNNPACK ...")
    lowered = lower_to_executorch_xnnpack(
        adapter=lowering_adapter,
        example_inputs=example_inputs,
        external_data_name=external_data_name,
        quantized=arguments.precision == "int8",
    )
    program_bytes = lowered.program_bytes
    program_file_name = "model-xnnpack.pte"
    (artifact_directory / program_file_name).write_bytes(program_bytes)
    print(
        f"Wrote {program_file_name}:",
        f"{len(program_bytes) / (1 << 20):.1f} MiB",
    )
    if external_data_name is not None:
        lowered.write_data_files(artifact_directory)
        data_path = artifact_directory / f"{external_data_name}.ptd"
        program_data_files = (data_path.name,)
        print(
            f"Wrote {data_path.name}:",
            f"{data_path.stat().st_size / (1 << 20):.1f} MiB",
        )

    largest_difference = 0.0
    if arguments.precision == "int8":
        # The Python runtime lacks the quantized kernels (embedding_byte),
        # so the runtime parity gate for int8 artifacts is the C++ test
        # suite, which executes the program against the recorded fixtures.
        print(
            "Runtime parity: gated by the C++ suite for int8 artifacts "
            "(fixtures record the quantized eager twin)."
        )
    for fixture, eager_outputs in zip(
        fixtures,
        eager_outputs_by_fixture,
        strict=True,
    ):
        if arguments.precision == "int8":
            break
        fixture_inputs = dict(fixture.input_tensors)
        runtime_outputs = run_executorch_program(
            program_bytes=program_bytes,
            inputs=tuple(tensor for _, tensor in fixture.input_tensors),
            data_path=data_path,
        )
        difference = maximum_task_probability_difference(
            schema=tagger.schema,
            reference_outputs=_decodable_flat_outputs(
                eager_outputs,
                tagger=tagger,
                calibrated=calibration is not None,
            ),
            candidate_outputs=_decodable_flat_outputs(
                runtime_outputs,
                tagger=tagger,
                calibrated=calibration is not None,
            ),
            token_mask=fixture_inputs["token_mask"],
        )
        runtime_decoded = decoded_sentence_predictions(
            schema=tagger.schema,
            logits=token_task_logits_from_flat_outputs(
                _decodable_flat_outputs(
                    runtime_outputs,
                    tagger=tagger,
                    calibrated=calibration is not None,
                ),
                schema=tagger.schema,
            ),
            token_mask=fixture_inputs["token_mask"],
            sentence_tokens=tuple(
                tuple(token["form"] for token in sentence["tokens"])
                for sentence in fixture.decoded_sentences
            ),
        )
        if tuple(runtime_decoded) != fixture.decoded_sentences:
            raise SystemExit(
                f"Runtime parity failed: {fixture.name} decodes differently "
                "in the ExecuTorch runtime than in eager PyTorch."
            )
        largest_difference = max(largest_difference, difference)
        print(
            f"Runtime parity {fixture.name}:",
            f"max probability |Δ| = {difference:.2e},",
            "decoded predictions identical",
        )
    if largest_difference > arguments.parity_tolerance:
        raise SystemExit(
            f"Runtime parity failed: {largest_difference:.2e} exceeds the "
            f"probability tolerance {arguments.parity_tolerance:.2e}."
        )

    tokenizer_contract = write_vocabulary_file(
        tokenizer=tagger.tokenizer,
        artifact_directory=artifact_directory,
    )
    write_json_file(
        artifact_directory / "labels.json",
        build_labels_payload(
            schema=tagger.schema,
            character_vocabulary=tagger.character_vocabulary,
            maximum_character_count=(
                None
                if tagger.character_vocabulary is None
                else tagger.maximum_character_count
            ),
        ),
    )
    write_json_file(
        artifact_directory / "fixtures.json",
        build_fixtures_payload(
            fixtures,
            # int8 fixtures record the quantized eager twin; the XNNPACK
            # int8 kernels differ from the twin by up to ~3e-2 in
            # calibrated probability (measured), so the recorded runtime
            # tolerance is widened accordingly.
            probability_tolerance=(
                max(arguments.parity_tolerance, 5e-2)
                if arguments.precision == "int8"
                else arguments.parity_tolerance
            ),
            lemma_top_k=arguments.fixture_lemma_top_k,
        ),
    )

    backbone = BackboneProvenance(
        model_id=str(tagger.checkpoint["backbone_model_id"]),
        revision=str(tagger.checkpoint["backbone_revision"]),
    )
    treebanks = tuple(
        TreebankProvenance(
            repository_id=profile.gold_treebank.repository_id,
            revision=profile.gold_treebank.revision,
            license_id=profile.gold_treebank.license_id,
        )
        for profile in profiles
    )
    small_program_entries: list[ModelProgramEntry] = []
    for small_subword_count, small_token_count in arguments.small_shapes or ():
        small_shapes = FixedExportShapes(
            batch_size=shapes.batch_size,
            subword_count=small_subword_count,
            token_count=small_token_count,
            character_count=shapes.character_count,
        )
        print(
            "Lowering small program:",
            f"subwords={small_shapes.subword_count}",
            f"tokens={small_shapes.token_count}",
        )
        small_sentences = _fitting_fixture_sentences(
            sentences=_pretokenized_development_sentences(
                profiles[0].gold_treebank.development_path,
                tagger,
            ),
            tokenizer=tagger.tokenizer,
            shapes=small_shapes,
            fixture_sentence_count=arguments.fixture_sentence_count,
        )
        small_inputs = build_fixture_input_tensors(
            sentences=small_sentences,
            tagger=tagger,
            shapes=small_shapes,
        )
        small_eager = None
        if arguments.precision != "int8":
            # The int8 twin is shape-guarded to the main shapes; the gates
            # below are skipped for int8, so no reference is needed here.
            with torch.no_grad():
                small_eager = tuple(
                    output.detach().clone()
                    for output in adapter(*(tensor for _, tensor in small_inputs))
                )
        small_lowered = lower_to_executorch_xnnpack(
            adapter=lowering_adapter,
            example_inputs=tuple(tensor for _, tensor in small_inputs),
            external_data_name=external_data_name,
            quantized=arguments.precision == "int8",
        )
        small_bytes = small_lowered.program_bytes
        small_file_name = (
            f"model-xnnpack-{small_shapes.subword_count}x{small_shapes.token_count}.pte"
        )
        (artifact_directory / small_file_name).write_bytes(small_bytes)
        print(f"Wrote {small_file_name}: {len(small_bytes) / (1 << 20):.1f} MiB")
        calibrated = calibration is not None
        small_difference = 0.0
        if arguments.precision == "int8":
            print("Runtime parity small program: gated by the C++ suite.")
        else:
            # The small program references the shared data file by content
            # hashes; executing it against the main program's model.ptd is
            # the gate proving weights are byte-identical across shapes.
            small_runtime = run_executorch_program(
                program_bytes=small_bytes,
                inputs=tuple(tensor for _, tensor in small_inputs),
                data_path=data_path,
            )
            small_difference = maximum_task_probability_difference(
                schema=tagger.schema,
                reference_outputs=_decodable_flat_outputs(
                    small_eager, tagger=tagger, calibrated=calibrated
                ),
                candidate_outputs=_decodable_flat_outputs(
                    small_runtime, tagger=tagger, calibrated=calibrated
                ),
                token_mask=dict(small_inputs)["token_mask"],
            )
            small_batch_tokens = tuple(
                sentence.tokens
                for sentence in repeat_pad_sentences(
                    small_sentences, batch_size=small_shapes.batch_size
                )
            )
            small_eager_decoded = decoded_sentence_predictions(
                schema=tagger.schema,
                logits=token_task_logits_from_flat_outputs(
                    _decodable_flat_outputs(
                        small_eager, tagger=tagger, calibrated=calibrated
                    ),
                    schema=tagger.schema,
                ),
                token_mask=dict(small_inputs)["token_mask"],
                sentence_tokens=small_batch_tokens,
            )
            small_runtime_decoded = decoded_sentence_predictions(
                schema=tagger.schema,
                logits=token_task_logits_from_flat_outputs(
                    _decodable_flat_outputs(
                        small_runtime, tagger=tagger, calibrated=calibrated
                    ),
                    schema=tagger.schema,
                ),
                token_mask=dict(small_inputs)["token_mask"],
                sentence_tokens=small_batch_tokens,
            )
            if tuple(small_runtime_decoded) != tuple(small_eager_decoded):
                raise SystemExit(
                    "Small-program decoded predictions diverge from eager output."
                )
            if small_difference > arguments.parity_tolerance:
                raise SystemExit(
                    "Small-program parity difference "
                    f"{small_difference} exceeds {arguments.parity_tolerance}."
                )
            print(
                "Runtime parity small program:",
                f"max probability |\u0394| = {small_difference:.2e},",
                "decoded predictions identical",
            )
        small_program_entries.append(
            ModelProgramEntry(
                file_name=small_file_name,
                format="executorch-pte",
                backend="xnnpack",
                precision=arguments.precision,
                sha256=sha256_of_bytes(small_bytes),
                size_bytes=len(small_bytes),
                shapes=small_shapes,
                inputs=_tensor_specs(small_inputs),
                output_names=_output_names(tagger, calibrated=calibrated),
                parity_maximum_probability_difference=small_difference,
                data_files=program_data_files,
            )
        )

    calibration_file_name = None
    if arguments.calibration_path is not None:
        calibration_file_name = "calibration.json"
        shutil.copyfile(
            arguments.calibration_path,
            artifact_directory / calibration_file_name,
        )
    manifest = ModelArtifactManifest(
        artifact_name=artifact_name,
        artifact_version=arguments.artifact_version,
        language_tags=tuple(profile.language_tag for profile in profiles),
        tasks=ARTIFACT_TASKS,
        labels_file="labels.json",
        vocabulary_file=tokenizer_contract.file_name,
        fixtures_file="fixtures.json",
        character_unicode_normalization="NFC",
        tokenizer=tokenizer_contract,
        programs=(
            ModelProgramEntry(
                file_name=program_file_name,
                format="executorch-pte",
                backend="xnnpack",
                precision=arguments.precision,
                sha256=sha256_of_bytes(program_bytes),
                size_bytes=len(program_bytes),
                shapes=shapes,
                inputs=_tensor_specs(fixtures[0].input_tensors),
                output_names=_output_names(
                    tagger,
                    calibrated=calibration is not None,
                ),
                parity_maximum_probability_difference=largest_difference,
                data_files=program_data_files,
            ),
            *small_program_entries,
        ),
        data_files=(
            ()
            if data_path is None
            else (
                ModelDataFileEntry(
                    file_name=data_path.name,
                    sha256=sha256_of_file(data_path),
                    size_bytes=data_path.stat().st_size,
                ),
            )
        ),
        checkpoint=CheckpointProvenance(
            run_name=arguments.checkpoint_path.parent.name,
            file_name=arguments.checkpoint_path.name,
            epoch_index=tagger.epoch_index,
            sha256=sha256_of_file(arguments.checkpoint_path),
            morphology_logit_correction_strength=(
                arguments.morphology_logit_correction_strength
            ),
        ),
        backbone=backbone,
        treebanks=treebanks,
        quantization="none",
        calibration_file=calibration_file_name,
    )
    write_json_file(artifact_directory / "manifest.json", manifest.to_json_dict())
    write_licenses_directory(
        artifact_directory=artifact_directory,
        backbone=backbone,
        treebanks=treebanks,
        silver_corpus_paths=_silver_corpus_paths(tagger.checkpoint),
    )

    print()
    print("Artifact:", artifact_directory)
    for entry in sorted(artifact_directory.rglob("*")):
        if entry.is_file():
            size = entry.stat().st_size
            print(f"  {entry.relative_to(artifact_directory)}  {size:,} bytes")


if __name__ == "__main__":
    main()
