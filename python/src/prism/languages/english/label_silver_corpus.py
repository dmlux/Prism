"""Label a prepared English silver corpus with the calibrated Teacher.

The command consumes a deterministic prefix of a prepared silver corpus,
runs the accepted labeler Teacher (and optionally a second agreement
Teacher), and writes calibrated soft labels as versioned shards plus a
provenance manifest. Filtering policies are deliberately not applied here;
they belong to the later training run.
"""

import argparse
import itertools
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from prism.data import (
    PretokenizedSilverSentence,
    iter_pretokenized_silver_sentences,
    load_silver_corpus_manifest,
    sha256_file,
)
from prism.languages.english.checkpoint_loading import (
    load_english_token_tagger,
)
from prism.training import (
    SILVER_LABEL_FORMAT_VERSION,
    SilverLabelManifest,
    SilverLabelShardReference,
    SilverSentenceLabels,
    generate_silver_labels,
    load_task_temperature_calibration,
    morphology_logit_correction_from_checkpoint,
    write_silver_label_manifest,
    write_silver_label_shard,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SilverLabelingArguments:
    silver_corpus_path: Path
    silver_manifest_path: Path
    checkpoint_path: Path
    calibration_path: Path
    output_directory: Path
    morphology_logit_correction_strength: float
    agreement_checkpoint_path: Path | None
    agreement_logit_correction_strength: float | None
    maximum_token_budget: int | None
    batch_size: int
    shard_sentence_count: int
    lemma_top_k: int
    device: str
    treebank_release: str


def parse_labeling_arguments(
    arguments: Sequence[str] | None = None,
) -> SilverLabelingArguments:
    parser = argparse.ArgumentParser(
        description="Write calibrated Teacher labels for a silver corpus.",
    )
    parser.add_argument(
        "--silver-corpus",
        type=Path,
        required=True,
        dest="silver_corpus_path",
    )
    parser.add_argument(
        "--silver-manifest",
        type=Path,
        required=True,
        dest="silver_manifest_path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        dest="checkpoint_path",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        dest="calibration_path",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--morphology-logit-correction-strength",
        type=float,
        default=0.0,
        help="Must match the strength recorded in the calibration artifact.",
    )
    parser.add_argument(
        "--agreement-checkpoint",
        type=Path,
        default=None,
        dest="agreement_checkpoint_path",
        help="Optional second Teacher whose decoded predictions are stored.",
    )
    parser.add_argument(
        "--agreement-logit-correction-strength",
        type=float,
        default=None,
        help="Required with --agreement-checkpoint.",
    )
    parser.add_argument(
        "--maximum-token-budget",
        type=int,
        default=None,
        help=(
            "Deterministic pilot budget: corpus-order sentences are labeled "
            "until the running token total reaches this budget; the crossing "
            "sentence is included. Omit to label the complete corpus."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-sentence-count", type=int, default=20000)
    parser.add_argument("--lemma-top-k", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parsed = parser.parse_args(arguments)
    if parsed.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if parsed.shard_sentence_count <= 0:
        parser.error("--shard-sentence-count must be greater than zero")
    if parsed.lemma_top_k <= 0:
        parser.error("--lemma-top-k must be greater than zero")
    if parsed.maximum_token_budget is not None and parsed.maximum_token_budget <= 0:
        parser.error("--maximum-token-budget must be greater than zero")
    if (parsed.agreement_checkpoint_path is None) != (
        parsed.agreement_logit_correction_strength is None
    ):
        parser.error(
            "--agreement-checkpoint and --agreement-logit-correction-strength "
            "must be provided together"
        )
    return SilverLabelingArguments(
        silver_corpus_path=parsed.silver_corpus_path,
        silver_manifest_path=parsed.silver_manifest_path,
        checkpoint_path=parsed.checkpoint_path,
        calibration_path=parsed.calibration_path,
        output_directory=parsed.output_directory,
        morphology_logit_correction_strength=(
            parsed.morphology_logit_correction_strength
        ),
        agreement_checkpoint_path=parsed.agreement_checkpoint_path,
        agreement_logit_correction_strength=(
            parsed.agreement_logit_correction_strength
        ),
        maximum_token_budget=parsed.maximum_token_budget,
        batch_size=parsed.batch_size,
        shard_sentence_count=parsed.shard_sentence_count,
        lemma_top_k=parsed.lemma_top_k,
        device=parsed.device,
        treebank_release=parsed.treebank_release,
    )


def _budgeted_sentences(
    sentences: Iterator[PretokenizedSilverSentence],
    maximum_token_budget: int | None,
) -> Iterator[PretokenizedSilverSentence]:
    consumed_tokens = 0
    for sentence in sentences:
        if (
            maximum_token_budget is not None
            and consumed_tokens >= maximum_token_budget
        ):
            return
        consumed_tokens += len(sentence.model_input.tokens)
        yield sentence


_PROGRESS_SENTENCE_INTERVAL = 1000


def _report_labeling_progress(
    records: Iterator[SilverSentenceLabels],
    *,
    maximum_token_budget: int | None,
) -> Iterator[SilverSentenceLabels]:
    sentence_count = 0
    token_count = 0
    for record in records:
        sentence_count += 1
        token_count += record.token_count
        if sentence_count % _PROGRESS_SENTENCE_INTERVAL == 0:
            if maximum_token_budget is None:
                print(
                    f"Labeling: {sentence_count} sentences,",
                    f"{token_count} tokens",
                    flush=True,
                )
            else:
                percent = min(100.0 * token_count / maximum_token_budget, 100.0)
                print(
                    f"Labeling: {sentence_count} sentences,",
                    f"{token_count}/{maximum_token_budget} tokens",
                    f"({percent:.1f}%)",
                    flush=True,
                )
        yield record


def main() -> None:
    arguments = parse_labeling_arguments()

    corpus_manifest = load_silver_corpus_manifest(arguments.silver_manifest_path)
    calibration = load_task_temperature_calibration(arguments.calibration_path)
    if calibration.checkpoint_path != str(arguments.checkpoint_path):
        raise ValueError(
            "Calibration artifact references a different checkpoint: "
            f"{calibration.checkpoint_path!r}"
        )
    if calibration.treebank_release != arguments.treebank_release:
        raise ValueError(
            "Calibration treebank release does not match the requested release."
        )
    if (
        calibration.morphology_logit_correction_strength
        != arguments.morphology_logit_correction_strength
    ):
        raise ValueError(
            "Calibration was fitted for correction strength "
            f"{calibration.morphology_logit_correction_strength}, not "
            f"{arguments.morphology_logit_correction_strength}."
        )

    labeler = load_english_token_tagger(
        checkpoint_path=arguments.checkpoint_path,
        required_language_tags=(corpus_manifest.language_tag,),
        treebank_release=arguments.treebank_release,
    )
    labeler_correction = morphology_logit_correction_from_checkpoint(
        labeler.checkpoint,
        strength=arguments.morphology_logit_correction_strength,
    )

    agreement = None
    agreement_correction = None
    if arguments.agreement_checkpoint_path is not None:
        agreement = load_english_token_tagger(
            checkpoint_path=arguments.agreement_checkpoint_path,
            required_language_tags=(corpus_manifest.language_tag,),
            treebank_release=arguments.treebank_release,
        )
        if labeler.checkpoint["schema"] != agreement.checkpoint["schema"]:
            raise ValueError(
                "Agreement Teacher must share the labeler's task schema."
            )
        assert arguments.agreement_logit_correction_strength is not None
        agreement_correction = morphology_logit_correction_from_checkpoint(
            agreement.checkpoint,
            strength=arguments.agreement_logit_correction_strength,
        )

    print("Silver corpus:", corpus_manifest.corpus_id)
    print("Language tag:", corpus_manifest.language_tag)
    print("Labeler epoch:", labeler.epoch_index + 1)
    print(
        "Agreement Teacher:",
        "absent"
        if arguments.agreement_checkpoint_path is None
        else arguments.agreement_checkpoint_path,
    )
    print(
        "Token budget:",
        "complete corpus"
        if arguments.maximum_token_budget is None
        else arguments.maximum_token_budget,
    )

    device = torch.device(arguments.device)
    labeled_records = generate_silver_labels(
        labeler=labeler.model,
        tokenizer=labeler.tokenizer,
        morphology_schema=labeler.schema.morphology,
        calibration=calibration,
        labeler_correction=labeler_correction,
        sentences=_budgeted_sentences(
            iter_pretokenized_silver_sentences(arguments.silver_corpus_path),
            arguments.maximum_token_budget,
        ),
        device=device,
        batch_size=arguments.batch_size,
        lemma_top_k=arguments.lemma_top_k,
        character_vocabulary=labeler.character_vocabulary,
        maximum_character_count=labeler.maximum_character_count,
        agreement_model=None if agreement is None else agreement.model,
        agreement_correction=agreement_correction,
    )
    records = _report_labeling_progress(
        labeled_records,
        maximum_token_budget=arguments.maximum_token_budget,
    )

    shard_references: list[SilverLabelShardReference] = []
    shard_index = 0
    while True:
        shard_records = tuple(
            itertools.islice(records, arguments.shard_sentence_count)
        )
        if not shard_records:
            break
        shard_index += 1
        shard_path = arguments.output_directory / f"labels-{shard_index:05d}.pt"
        reference = write_silver_label_shard(
            records=shard_records,
            path=shard_path,
        )
        shard_references.append(reference)
        print(
            f"Shard {reference.file_name}:",
            f"{reference.sentence_count} sentences,",
            f"{reference.token_count} tokens",
            flush=True,
        )

    if not shard_references:
        raise ValueError("Silver labeling produced no sentences.")

    manifest = SilverLabelManifest(
        format_version=SILVER_LABEL_FORMAT_VERSION,
        corpus_path=str(arguments.silver_corpus_path),
        corpus_sha256=sha256_file(arguments.silver_corpus_path),
        corpus_manifest=json.loads(
            arguments.silver_manifest_path.read_text(encoding="utf-8")
        ),
        labeler_checkpoint_path=str(arguments.checkpoint_path),
        labeler_checkpoint_sha256=sha256_file(arguments.checkpoint_path),
        labeler_epoch_index=labeler.epoch_index,
        calibration=json.loads(
            arguments.calibration_path.read_text(encoding="utf-8")
        ),
        morphology_logit_correction_strength=(
            arguments.morphology_logit_correction_strength
        ),
        agreement_checkpoint_path=(
            None
            if arguments.agreement_checkpoint_path is None
            else str(arguments.agreement_checkpoint_path)
        ),
        agreement_checkpoint_sha256=(
            None
            if arguments.agreement_checkpoint_path is None
            else sha256_file(arguments.agreement_checkpoint_path)
        ),
        agreement_logit_correction_strength=(
            arguments.agreement_logit_correction_strength
        ),
        lemma_top_k=arguments.lemma_top_k,
        probability_dtype="float16",
        maximum_token_budget=arguments.maximum_token_budget,
        sentence_count=sum(
            reference.sentence_count for reference in shard_references
        ),
        token_count=sum(reference.token_count for reference in shard_references),
        shards=tuple(shard_references),
    )
    manifest_path = arguments.output_directory / "labels-manifest.json"
    write_silver_label_manifest(manifest, manifest_path)

    print()
    print("Sentences:", manifest.sentence_count)
    print("Tokens:", manifest.token_count)
    print("Shards:", len(manifest.shards))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
