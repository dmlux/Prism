"""Versioned model-artifact directory contract.

A released language package is a directory of portable model programs plus
the deterministic data a native runtime needs to reproduce Python decoding:
a manifest, the label schema, the subword vocabulary, and recorded parity
fixtures. This module owns the typed manifest and the JSON serialization so
Python export and future Swift, Java/Kotlin, and C++ readers share one
documented contract instead of loosely structured dictionaries.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from prism.exporting.shapes import FixedExportShapes
from prism.modeling.decoding import (
    decode_token_task_logits,
    morphology_label_scores,
)
from prism.modeling.outputs import TokenTaskLogits
from prism.schema import (
    CharacterVocabularySchema,
    TokenTaskSchema,
)
from prism.schema.morphology import NO_MORPHOLOGY_VALUE
from prism.schema.serialization import (
    serialize_character_vocabulary_schema,
    serialize_token_task_schema,
)


ARTIFACT_MANIFEST_FORMAT_VERSION = 1
ARTIFACT_LABELS_FORMAT_VERSION = 1
ARTIFACT_FIXTURES_FORMAT_VERSION = 1

INVALID_LEMMA_RULE_LEMMA = "<INVALID_LEMMA_RULE>"

_TENSOR_DTYPES: dict[str, torch.dtype] = {
    "int64": torch.int64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bool": torch.bool,
}


def _require_non_empty(value: str, description: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(
            f"{description} must be non-empty and have no surrounding whitespace."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TreebankProvenance:
    repository_id: str
    revision: str
    license_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.repository_id, "Treebank repository ID")
        _require_non_empty(self.revision, "Treebank revision")
        _require_non_empty(self.license_id, "Treebank license ID")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "repository_id": self.repository_id,
            "revision": self.revision,
            "license_id": self.license_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class BackboneProvenance:
    model_id: str
    revision: str

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "Backbone model ID")
        _require_non_empty(self.revision, "Backbone revision")

    def to_json_dict(self) -> dict[str, object]:
        return {"model_id": self.model_id, "revision": self.revision}


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointProvenance:
    run_name: str
    file_name: str
    epoch_index: int
    sha256: str
    morphology_logit_correction_strength: float

    def __post_init__(self) -> None:
        _require_non_empty(self.run_name, "Checkpoint run name")
        _require_non_empty(self.file_name, "Checkpoint file name")
        _require_non_empty(self.sha256, "Checkpoint digest")
        if self.epoch_index < 0:
            raise ValueError("Checkpoint epoch index must not be negative.")
        if not 0.0 <= self.morphology_logit_correction_strength <= 1.0:
            raise ValueError(
                "Morphology logit-correction strength must be between zero and one."
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "run_name": self.run_name,
            "file_name": self.file_name,
            "epoch_index": self.epoch_index,
            "sha256": self.sha256,
            "morphology_logit_correction_strength": (
                self.morphology_logit_correction_strength
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizerContract:
    file_name: str
    class_name: str
    padding_token_id: int

    def __post_init__(self) -> None:
        _require_non_empty(self.file_name, "Tokenizer file name")
        _require_non_empty(self.class_name, "Tokenizer class name")
        if self.padding_token_id < 0:
            raise ValueError("Tokenizer padding token ID must not be negative.")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "file_name": self.file_name,
            "class_name": self.class_name,
            "padding_token_id": self.padding_token_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "Tensor name")
        if self.dtype not in _TENSOR_DTYPES:
            raise ValueError(f"Unsupported tensor dtype: {self.dtype!r}.")
        if not self.shape or any(size < 1 for size in self.shape):
            raise ValueError("Tensor shapes must contain positive sizes.")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProgramEntry:
    """One lowered model program inside the artifact directory."""

    file_name: str
    format: str
    backend: str
    precision: str
    sha256: str
    size_bytes: int
    shapes: FixedExportShapes
    inputs: tuple[TensorSpec, ...]
    output_names: tuple[str, ...]
    parity_maximum_probability_difference: float

    def __post_init__(self) -> None:
        _require_non_empty(self.file_name, "Model program file name")
        _require_non_empty(self.format, "Model program format")
        _require_non_empty(self.backend, "Model program backend")
        _require_non_empty(self.precision, "Model program precision")
        _require_non_empty(self.sha256, "Model program digest")
        if self.size_bytes < 1:
            raise ValueError("Model program size must be positive.")
        if not self.inputs:
            raise ValueError("Model program inputs must not be empty.")
        if not self.output_names:
            raise ValueError("Model program output names must not be empty.")
        if self.parity_maximum_probability_difference < 0.0:
            raise ValueError(
                "Parity maximum probability difference must not be negative."
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "file_name": self.file_name,
            "format": self.format,
            "backend": self.backend,
            "precision": self.precision,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "shapes": {
                "batch_size": self.shapes.batch_size,
                "subword_count": self.shapes.subword_count,
                "token_count": self.shapes.token_count,
                "character_count": self.shapes.character_count,
            },
            "inputs": [spec.to_json_dict() for spec in self.inputs],
            "output_names": list(self.output_names),
            "parity_maximum_probability_difference": (
                self.parity_maximum_probability_difference
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelArtifactManifest:
    artifact_name: str
    artifact_version: str
    language_tags: tuple[str, ...]
    tasks: tuple[str, ...]
    labels_file: str
    vocabulary_file: str
    fixtures_file: str
    character_unicode_normalization: str
    tokenizer: TokenizerContract
    programs: tuple[ModelProgramEntry, ...]
    checkpoint: CheckpointProvenance
    backbone: BackboneProvenance
    treebanks: tuple[TreebankProvenance, ...]
    quantization: str
    calibration_file: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_name, "Artifact name")
        _require_non_empty(self.artifact_version, "Artifact version")
        if not self.language_tags:
            raise ValueError("Artifact language tags must not be empty.")
        if not self.tasks:
            raise ValueError("Artifact tasks must not be empty.")
        if not self.programs:
            raise ValueError("Artifact must contain at least one model program.")
        if not self.treebanks:
            raise ValueError("Artifact treebank provenance must not be empty.")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "manifest_format_version": ARTIFACT_MANIFEST_FORMAT_VERSION,
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "language_tags": list(self.language_tags),
            "tasks": list(self.tasks),
            "labels_file": self.labels_file,
            "vocabulary_file": self.vocabulary_file,
            "fixtures_file": self.fixtures_file,
            "character_unicode_normalization": (self.character_unicode_normalization),
            "tokenizer": self.tokenizer.to_json_dict(),
            "programs": [program.to_json_dict() for program in self.programs],
            "checkpoint": self.checkpoint.to_json_dict(),
            "backbone": self.backbone.to_json_dict(),
            "treebanks": [treebank.to_json_dict() for treebank in self.treebanks],
            "quantization": self.quantization,
            "calibration_file": self.calibration_file,
        }


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_labels_payload(
    *,
    schema: TokenTaskSchema,
    character_vocabulary: CharacterVocabularySchema | None,
    maximum_character_count: int | None,
) -> dict[str, object]:
    if (character_vocabulary is None) != (maximum_character_count is None):
        raise ValueError(
            "Character vocabulary and maximum character count must be "
            "provided together."
        )

    return {
        "labels_format_version": ARTIFACT_LABELS_FORMAT_VERSION,
        "schema": serialize_token_task_schema(schema),
        "character_vocabulary": (
            None
            if character_vocabulary is None
            else serialize_character_vocabulary_schema(character_vocabulary)
        ),
        "maximum_character_count": maximum_character_count,
    }


def serialize_tensor(name: str, tensor: Tensor) -> dict[str, object]:
    dtype_names = {dtype: name for name, dtype in _TENSOR_DTYPES.items()}
    dtype_name = dtype_names.get(tensor.dtype)
    if dtype_name is None:
        raise ValueError(f"Unsupported fixture tensor dtype: {tensor.dtype}.")

    return {
        "name": name,
        "dtype": dtype_name,
        "shape": list(tensor.shape),
        "data": tensor.reshape(-1).tolist(),
    }


def deserialize_tensor(payload: dict[str, object]) -> tuple[str, Tensor]:
    name = payload.get("name")
    dtype_name = payload.get("dtype")
    shape = payload.get("shape")
    data = payload.get("data")
    if not isinstance(name, str):
        raise ValueError("Fixture tensor name must be a string.")
    if not isinstance(dtype_name, str) or dtype_name not in _TENSOR_DTYPES:
        raise ValueError(f"Unsupported fixture tensor dtype: {dtype_name!r}.")
    if not isinstance(shape, list):
        raise ValueError("Fixture tensor shape must be a list.")
    if not isinstance(data, list):
        raise ValueError("Fixture tensor data must be a list.")

    tensor = torch.tensor(data, dtype=_TENSOR_DTYPES[dtype_name]).reshape(shape)
    return name, tensor


def token_task_logits_from_flat_outputs(
    outputs: Sequence[Tensor],
    *,
    schema: TokenTaskSchema,
) -> TokenTaskLogits:
    """Rebuild typed task logits from the flat export output tuple."""

    feature_count = len(schema.morphology.features)
    if len(outputs) != feature_count + 2:
        raise ValueError(
            f"Export adapter returned {len(outputs)} outputs; expected "
            f"{feature_count + 2}."
        )
    return TokenTaskLogits(
        upos_logits=outputs[0],
        morphology_logits=tuple(outputs[1 : 1 + feature_count]),
        lemma_rule_logits=outputs[-1],
    )


def task_probability_tensors(
    *,
    schema: TokenTaskSchema,
    logits: TokenTaskLogits,
) -> tuple[Tensor, ...]:
    """Convert task logits into the probabilities native runtimes consume.

    Raw logits are a poor parity target: the bundle reranker maps saturated
    candidate probabilities through ``log``/``logit``, so numerically
    irrelevant runtime differences near the epsilon clamp inflate into large
    logit gaps. Probabilities compare what decoding and confidence actually
    use.
    """

    probabilities: list[Tensor] = [torch.softmax(logits.upos_logits, dim=-1)]
    for feature_logits, feature in zip(
        logits.morphology_logits,
        schema.morphology.features,
        strict=True,
    ):
        probabilities.append(
            morphology_label_scores(
                feature_logits=feature_logits,
                feature_schema=feature,
            )
        )
    probabilities.append(torch.softmax(logits.lemma_rule_logits, dim=-1))
    return tuple(probabilities)


def maximum_task_probability_difference(
    *,
    schema: TokenTaskSchema,
    reference_outputs: Sequence[Tensor],
    candidate_outputs: Sequence[Tensor],
    token_mask: Tensor,
) -> float:
    """Compare two flat output tuples in probability space at valid tokens.

    Outputs at padded positions are contractually undefined, so only
    positions inside the token mask participate.
    """

    reference_probabilities = task_probability_tensors(
        schema=schema,
        logits=token_task_logits_from_flat_outputs(
            reference_outputs,
            schema=schema,
        ),
    )
    candidate_probabilities = task_probability_tensors(
        schema=schema,
        logits=token_task_logits_from_flat_outputs(
            candidate_outputs,
            schema=schema,
        ),
    )

    largest = 0.0
    for reference, candidate in zip(
        reference_probabilities,
        candidate_probabilities,
        strict=True,
    ):
        if reference.shape != candidate.shape:
            raise ValueError(
                "Parity comparison requires matching output shapes: "
                f"{tuple(reference.shape)} versus {tuple(candidate.shape)}."
            )
        mask = token_mask.unsqueeze(-1).expand_as(reference)
        difference = (reference - candidate).abs()[mask].max().item()
        largest = max(largest, float(difference))

    return largest


def decoded_sentence_predictions(
    *,
    schema: TokenTaskSchema,
    logits: TokenTaskLogits,
    token_mask: Tensor,
    sentence_tokens: Sequence[Sequence[str]],
) -> list[dict[str, object]]:
    """Decode canonical per-token predictions into a JSON-friendly payload.

    The payload records the same argmax/threshold decisions a native runtime
    must reproduce: UPOS label, active morphology values per feature (an
    empty list means no annotation), and the lemma from applying the
    predicted edit rule to the token form.
    """

    predictions = decode_token_task_logits(
        logits=logits,
        token_mask=token_mask,
        morphology_schema=schema.morphology,
    )

    sentences_payload: list[dict[str, object]] = []
    for sentence_index, tokens in enumerate(sentence_tokens):
        tokens_payload: list[dict[str, object]] = []
        for token_index, form in enumerate(tokens):
            if not bool(token_mask[sentence_index, token_index].item()):
                raise ValueError(
                    "Decoded fixture tokens must lie inside the token mask."
                )

            upos_id = int(predictions.upos_ids[sentence_index, token_index].item())
            morphology_payload: dict[str, list[str]] = {}
            for feature, feature_predictions in zip(
                schema.morphology.features,
                predictions.morphology_predictions,
                strict=True,
            ):
                active_labels = [
                    label
                    for label, is_active in zip(
                        feature.labels,
                        feature_predictions[sentence_index, token_index].tolist(),
                        strict=True,
                    )
                    if is_active and label != NO_MORPHOLOGY_VALUE
                ]
                morphology_payload[feature.name] = active_labels

            lemma_rule_id = int(
                predictions.lemma_rule_ids[sentence_index, token_index].item()
            )
            try:
                lemma = schema.lemma_rules.rule_for_id(lemma_rule_id).apply(form)
            except ValueError:
                # A globally shared edit-rule inventory can select a rule that
                # removes more characters than a short token contains. It is an
                # incorrect prediction the native runtime must reproduce.
                lemma = INVALID_LEMMA_RULE_LEMMA

            tokens_payload.append(
                {
                    "form": form,
                    "upos": schema.upos.labels[upos_id],
                    "morphology": morphology_payload,
                    "lemma_rule_id": lemma_rule_id,
                    "lemma": lemma,
                }
            )
        sentences_payload.append({"tokens": tokens_payload})

    return sentences_payload


@dataclass(frozen=True, slots=True, kw_only=True)
class ParityFixtureBatch:
    """One recorded batch a native runtime replays for numerical parity."""

    name: str
    input_tensors: tuple[tuple[str, Tensor], ...]
    expected_output_tensors: tuple[tuple[str, Tensor], ...]
    decoded_sentences: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "Fixture batch name")
        if not self.input_tensors:
            raise ValueError("Fixture batch inputs must not be empty.")
        if not self.expected_output_tensors:
            raise ValueError("Fixture batch outputs must not be empty.")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "inputs": [
                serialize_tensor(name, tensor) for name, tensor in self.input_tensors
            ],
            "expected_outputs": [
                serialize_tensor(name, tensor)
                for name, tensor in self.expected_output_tensors
            ],
            "decoded_sentences": list(self.decoded_sentences),
        }


def top_k_output_tensors(
    *,
    name: str,
    logits: Tensor,
    top_k: int,
) -> tuple[tuple[str, Tensor], tuple[str, Tensor]]:
    """Compress one wide logit head into recorded top-k ids and values.

    The lemma head spans thousands of edit rules, so storing its complete
    logits would dominate the fixture file. A runtime gathers its own logits
    at the recorded ids and compares the values.
    """

    if top_k < 1:
        raise ValueError("Fixture top-k must be positive.")

    values, ids = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
    return (
        (f"{name}:top_ids", ids),
        (f"{name}:top_values", values),
    )


def build_fixtures_payload(
    fixtures: Sequence[ParityFixtureBatch],
    *,
    probability_tolerance: float,
    lemma_top_k: int | None = None,
) -> dict[str, object]:
    if not fixtures:
        raise ValueError("Fixtures payload must contain fixture batches.")
    if probability_tolerance <= 0.0:
        raise ValueError("Fixture probability tolerance must be positive.")
    if lemma_top_k is not None and lemma_top_k < 1:
        raise ValueError("Fixture lemma top-k must be positive.")

    return {
        "fixtures_format_version": ARTIFACT_FIXTURES_FORMAT_VERSION,
        "comparison": {
            # Raw logits are recorded for debugging, but runtimes must
            # compare task probabilities: the bundle reranker inflates
            # numerically irrelevant differences of saturated probabilities
            # into large logit gaps.
            "space": "task-probabilities",
            "probability_tolerance": probability_tolerance,
            "outputs_valid_only_inside_token_mask": True,
            "decoded_sentences_must_match_exactly": True,
            "lemma_top_k": lemma_top_k,
        },
        "fixtures": [fixture.to_json_dict() for fixture in fixtures],
    }
