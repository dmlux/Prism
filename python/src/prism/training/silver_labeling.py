"""Offline, provenance-carrying teacher labels for silver corpora.

The label artifact stores the calibrated, corrected soft predictions of the
accepted labeler Teacher — plus, optionally, the decoded predictions of a
second agreement Teacher — for a deterministic prefix of a prepared silver
corpus. It deliberately stores raw predictions instead of already-filtered
labels: confidence thresholds, two-teacher agreement, and sentence-discard
policies are applied later at training time, so selection-policy ablations
never require an expensive relabeling run.

Per token the artifact keeps the complete calibrated UPOS distribution, the
complete calibrated distribution of every morphology feature, and the top-k
lemma-rule distribution, all stored as float16. Sentences are grouped into
`torch.save` shards; a JSON manifest records the source corpus, both teacher
checkpoints, the calibration artifact, counts, and shard checksums.
"""

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from prism.data.silver import PretokenizedSilverSentence, sha256_file
from prism.modeling import TokenizedBatch, tokenize_pretokenized_sentences
from prism.modeling.character_batches import encode_character_token_batch
from prism.modeling.decoding import (
    MorphologyLogitCorrection,
    apply_morphology_logit_correction,
    decode_token_task_logits,
)
from prism.modeling.outputs import TokenTaskLogits
from prism.schema import CharacterVocabularySchema, MorphologySchema
from prism.training.calibration import (
    TaskTemperatureCalibration,
    calibrated_task_probabilities,
)


SILVER_LABEL_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverSentenceLabels:
    """Calibrated teacher predictions for one silver sentence."""

    document_id: str
    sentence_index: int
    upos_probabilities: Tensor
    morphology_probabilities: tuple[Tensor, ...]
    lemma_rule_ids: Tensor
    lemma_rule_probabilities: Tensor
    agreement_upos_ids: Tensor | None
    agreement_lemma_rule_ids: Tensor | None
    agreement_morphology_predictions: tuple[Tensor, ...] | None

    def __post_init__(self) -> None:
        token_count = self.token_count
        if token_count == 0:
            raise ValueError("Silver sentence labels require tokens.")
        if any(
            probabilities.shape[0] != token_count
            for probabilities in self.morphology_probabilities
        ):
            raise ValueError("Morphology probabilities must cover every token.")
        if self.lemma_rule_ids.shape != self.lemma_rule_probabilities.shape:
            raise ValueError("Lemma rule IDs and probabilities must align.")
        if self.lemma_rule_ids.shape[0] != token_count:
            raise ValueError("Lemma rules must cover every token.")
        agreement_fields = (
            self.agreement_upos_ids,
            self.agreement_lemma_rule_ids,
            self.agreement_morphology_predictions,
        )
        if any(field is None for field in agreement_fields) != all(
            field is None for field in agreement_fields
        ):
            raise ValueError("Agreement predictions must be complete or absent.")

    @property
    def token_count(self) -> int:
        return int(self.upos_probabilities.shape[0])


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverLabelShardReference:
    file_name: str
    sentence_count: int
    token_count: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.file_name or "/" in self.file_name:
            raise ValueError("Shard file name must be a plain file name.")
        if self.sentence_count <= 0 or self.token_count < self.sentence_count:
            raise ValueError("Shard counts must cover its sentences.")
        if len(self.sha256) != 64:
            raise ValueError("Shard SHA-256 must be lowercase hex.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverLabelManifest:
    format_version: int
    corpus_path: str
    corpus_sha256: str
    corpus_manifest: Mapping[str, object]
    labeler_checkpoint_path: str
    labeler_checkpoint_sha256: str
    labeler_epoch_index: int
    calibration: Mapping[str, object]
    morphology_logit_correction_strength: float
    agreement_checkpoint_path: str | None
    agreement_checkpoint_sha256: str | None
    agreement_logit_correction_strength: float | None
    lemma_top_k: int
    probability_dtype: str
    maximum_token_budget: int | None
    sentence_count: int
    token_count: int
    shards: tuple[SilverLabelShardReference, ...]

    def __post_init__(self) -> None:
        if self.format_version != SILVER_LABEL_FORMAT_VERSION:
            raise ValueError("Unsupported silver-label format version.")
        if self.lemma_top_k <= 0:
            raise ValueError("Lemma top-k must be positive.")
        if self.probability_dtype != "float16":
            raise ValueError("Silver labels currently require float16 storage.")
        if not self.shards:
            raise ValueError("Silver-label manifest requires shards.")
        if self.sentence_count != sum(shard.sentence_count for shard in self.shards):
            raise ValueError("Manifest sentence count must match its shards.")
        if self.token_count != sum(shard.token_count for shard in self.shards):
            raise ValueError("Manifest token count must match its shards.")
        if (self.agreement_checkpoint_path is None) != (
            self.agreement_checkpoint_sha256 is None
        ) or (self.agreement_checkpoint_path is None) != (
            self.agreement_logit_correction_strength is None
        ):
            raise ValueError("Agreement metadata must be complete or absent.")
        if not math.isfinite(self.morphology_logit_correction_strength):
            raise ValueError("Logit correction strength must be finite.")


def _record_to_serializable(record: SilverSentenceLabels) -> dict[str, object]:
    return {
        "document_id": record.document_id,
        "sentence_index": record.sentence_index,
        "upos_probabilities": record.upos_probabilities,
        "morphology_probabilities": tuple(record.morphology_probabilities),
        "lemma_rule_ids": record.lemma_rule_ids,
        "lemma_rule_probabilities": record.lemma_rule_probabilities,
        "agreement_upos_ids": record.agreement_upos_ids,
        "agreement_lemma_rule_ids": record.agreement_lemma_rule_ids,
        "agreement_morphology_predictions": (
            None
            if record.agreement_morphology_predictions is None
            else tuple(record.agreement_morphology_predictions)
        ),
    }


def _record_from_serializable(value: dict[str, object]) -> SilverSentenceLabels:
    agreement_morphology = value["agreement_morphology_predictions"]
    return SilverSentenceLabels(
        document_id=value["document_id"],
        sentence_index=value["sentence_index"],
        upos_probabilities=value["upos_probabilities"],
        morphology_probabilities=tuple(value["morphology_probabilities"]),
        lemma_rule_ids=value["lemma_rule_ids"],
        lemma_rule_probabilities=value["lemma_rule_probabilities"],
        agreement_upos_ids=value["agreement_upos_ids"],
        agreement_lemma_rule_ids=value["agreement_lemma_rule_ids"],
        agreement_morphology_predictions=(
            None if agreement_morphology is None else tuple(agreement_morphology)
        ),
    )


def write_silver_label_shard(
    *,
    records: Sequence[SilverSentenceLabels],
    path: Path,
) -> SilverLabelShardReference:
    if not records:
        raise ValueError("Silver-label shards require records.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": SILVER_LABEL_FORMAT_VERSION,
            "records": [_record_to_serializable(record) for record in records],
        },
        temporary_path,
    )
    temporary_path.replace(path)
    return SilverLabelShardReference(
        file_name=path.name,
        sentence_count=len(records),
        token_count=sum(record.token_count for record in records),
        sha256=sha256_file(path),
    )


def iter_silver_label_records(
    *,
    directory: Path,
    manifest: SilverLabelManifest,
    verify_checksums: bool = False,
) -> Iterator[SilverSentenceLabels]:
    for shard in manifest.shards:
        shard_path = directory / shard.file_name
        if verify_checksums and sha256_file(shard_path) != shard.sha256:
            raise ValueError(f"Silver-label shard checksum mismatch: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != SILVER_LABEL_FORMAT_VERSION:
            raise ValueError(f"Unsupported silver-label shard format: {shard_path}")
        records = payload["records"]
        if len(records) != shard.sentence_count:
            raise ValueError(f"Silver-label shard record count mismatch: {shard_path}")
        for value in records:
            yield _record_from_serializable(value)


def write_silver_label_manifest(
    manifest: SilverLabelManifest,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_silver_label_manifest(path: Path) -> SilverLabelManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Silver-label manifest must contain a JSON object.")
    shards = value.pop("shards", None)
    if not isinstance(shards, list):
        raise ValueError("Silver-label manifest must contain shards.")
    return SilverLabelManifest(
        **value,
        shards=tuple(
            SilverLabelShardReference(**shard) for shard in shards
        ),
    )


def _batched(
    sentences: Iterable[PretokenizedSilverSentence],
    batch_size: int,
) -> Iterator[tuple[PretokenizedSilverSentence, ...]]:
    batch: list[PretokenizedSilverSentence] = []
    for sentence in sentences:
        batch.append(sentence)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _forward_labeler(
    *,
    model: nn.Module,
    model_inputs: TokenizedBatch,
    character_inputs,
) -> TokenTaskLogits:
    if getattr(model, "character_encoder", None) is None:
        logits = model(model_inputs)
    else:
        if character_inputs is None:
            raise ValueError("Character-aware labeler requires character inputs.")
        logits = model(model_inputs, character_inputs)
    if not isinstance(logits, TokenTaskLogits):
        raise TypeError("Labeler model must return TokenTaskLogits.")
    return logits


def generate_silver_labels(
    *,
    labeler: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    morphology_schema: MorphologySchema,
    calibration: TaskTemperatureCalibration,
    labeler_correction: MorphologyLogitCorrection | None,
    sentences: Iterable[PretokenizedSilverSentence],
    device: torch.device,
    batch_size: int,
    lemma_top_k: int,
    character_vocabulary: CharacterVocabularySchema | None,
    maximum_character_count: int,
    agreement_model: nn.Module | None = None,
    agreement_correction: MorphologyLogitCorrection | None = None,
) -> Iterator[SilverSentenceLabels]:
    """Yield calibrated labeler predictions for every silver sentence."""

    if batch_size <= 0:
        raise ValueError("Labeling batch size must be positive.")
    if lemma_top_k <= 0:
        raise ValueError("Lemma top-k must be positive.")

    labeler.to(device)
    labeler.eval()
    if agreement_model is not None:
        agreement_model.to(device)
        agreement_model.eval()

    for sentence_batch in _batched(sentences, batch_size):
        model_inputs = tokenize_pretokenized_sentences(
            tokenizer=tokenizer,
            sentences=tuple(sentence.model_input for sentence in sentence_batch),
        ).to(device)
        character_inputs = (
            None
            if character_vocabulary is None
            else encode_character_token_batch(
                token_sequences=tuple(
                    sentence.model_input.tokens for sentence in sentence_batch
                ),
                vocabulary=character_vocabulary,
                maximum_character_count=maximum_character_count,
            ).to(device)
        )

        with torch.inference_mode():
            logits = _forward_labeler(
                model=labeler,
                model_inputs=model_inputs,
                character_inputs=character_inputs,
            )
            corrected = (
                logits
                if labeler_correction is None
                else apply_morphology_logit_correction(
                    logits=logits,
                    morphology_schema=morphology_schema,
                    correction=labeler_correction,
                )
            )
            probabilities = calibrated_task_probabilities(
                logits=corrected,
                morphology_schema=morphology_schema,
                calibration=calibration,
            )
            top_probabilities, top_ids = probabilities.lemma_rule_probabilities.topk(
                min(lemma_top_k, probabilities.lemma_rule_probabilities.shape[-1]),
                dim=-1,
            )

            agreement_predictions = None
            if agreement_model is not None:
                agreement_logits = _forward_labeler(
                    model=agreement_model,
                    model_inputs=model_inputs,
                    character_inputs=character_inputs,
                )
                agreement_corrected = (
                    agreement_logits
                    if agreement_correction is None
                    else apply_morphology_logit_correction(
                        logits=agreement_logits,
                        morphology_schema=morphology_schema,
                        correction=agreement_correction,
                    )
                )
                agreement_predictions = decode_token_task_logits(
                    logits=agreement_corrected,
                    token_mask=model_inputs.token_mask,
                    morphology_schema=morphology_schema,
                )

        for position, sentence in enumerate(sentence_batch):
            token_count = len(sentence.model_input.tokens)
            yield SilverSentenceLabels(
                document_id=sentence.document_id,
                sentence_index=sentence.sentence_index,
                upos_probabilities=(
                    probabilities.upos_probabilities[position, :token_count]
                    .to(torch.float16)
                    .cpu()
                ),
                morphology_probabilities=tuple(
                    feature_probabilities[position, :token_count]
                    .to(torch.float16)
                    .cpu()
                    for feature_probabilities in (
                        probabilities.morphology_probabilities
                    )
                ),
                lemma_rule_ids=(
                    top_ids[position, :token_count].to(torch.int32).cpu()
                ),
                lemma_rule_probabilities=(
                    top_probabilities[position, :token_count]
                    .to(torch.float16)
                    .cpu()
                ),
                agreement_upos_ids=(
                    None
                    if agreement_predictions is None
                    else agreement_predictions.upos_ids[position, :token_count]
                    .to(torch.int16)
                    .cpu()
                ),
                agreement_lemma_rule_ids=(
                    None
                    if agreement_predictions is None
                    else agreement_predictions.lemma_rule_ids[
                        position, :token_count
                    ]
                    .to(torch.int32)
                    .cpu()
                ),
                agreement_morphology_predictions=(
                    None
                    if agreement_predictions is None
                    else tuple(
                        feature_predictions[position, :token_count].cpu()
                        for feature_predictions in (
                            agreement_predictions.morphology_predictions
                        )
                    )
                ),
            )
