"""Fit per-task-head temperatures for a Norwegian checkpoint.

Temperature scaling never changes argmax decisions; it only repairs the
overconfident probabilities of a trained checkpoint. The resulting versioned
calibration artifact is a prerequisite for confidence-filtered silver
labeling and, later, for shipped confidence values.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from prism.conllu import read_sentences
from prism.data import encode_norwegian_sentences
from prism.languages.norwegian import (
    norwegian_training_profiles_for_language_tag,
)
from prism.languages.norwegian.checkpoint_loading import (
    load_norwegian_token_tagger,
)
from prism.training import (
    CalibrationStatistics,
    TaskTemperatureCalibration,
    calibrate_task_heads,
    evaluate_supervised_token_task_step,
    iter_supervised_token_task_batches,
    morphology_logit_correction_from_checkpoint,
    write_task_temperature_calibration,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineCalibrationArguments:
    checkpoint_path: Path
    calibration_path: Path | None
    language_tag: str
    device: str
    treebank_release: str
    morphology_logit_correction_strength: float


def parse_calibration_arguments(
    arguments: Sequence[str] | None = None,
) -> BaselineCalibrationArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-task-head temperatures on the development split."
        ),
    )
    parser.add_argument(
        "--language-tag",
        choices=("nb", "nn", "no"),
        default="no",
        help=(
            "Development data used for the fit; the joint model calibrates "
            "on the combined Bokmål and Nynorsk splits by default."
        ),
    )
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
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
        default=None,
        dest="calibration_path",
        help="Output artifact path (default: calibration.json beside the checkpoint).",
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
            "Calibrate the corrected logits with this strength; it must match "
            "the strength that later consumes the calibration."
        ),
    )
    parsed = parser.parse_args(arguments)
    return BaselineCalibrationArguments(
        checkpoint_path=parsed.checkpoint_path,
        calibration_path=parsed.calibration_path,
        language_tag=parsed.language_tag,
        device=parsed.device,
        treebank_release=parsed.treebank_release,
        morphology_logit_correction_strength=(
            parsed.morphology_logit_correction_strength
        ),
    )


def main() -> None:
    arguments = parse_calibration_arguments()

    calibration_profiles = norwegian_training_profiles_for_language_tag(
        arguments.language_tag,
        treebank_release=arguments.treebank_release,
    )
    tagger = load_norwegian_token_tagger(
        checkpoint_path=arguments.checkpoint_path,
        required_language_tags=tuple(
            profile.language_tag for profile in calibration_profiles
        ),
        treebank_release=arguments.treebank_release,
    )
    morphology_logit_correction = morphology_logit_correction_from_checkpoint(
        tagger.checkpoint,
        strength=arguments.morphology_logit_correction_strength,
    )

    device = torch.device(arguments.device)
    tagger.model.to(device)

    print("Calibrating checkpoint epoch:", tagger.epoch_index + 1)
    print("Language tags:", ", ".join(p.language_tag for p in calibration_profiles))
    print(
        "Morphology logit correction:",
        f"{arguments.morphology_logit_correction_strength:.2f}",
    )

    statistics = CalibrationStatistics.empty(tagger.schema.morphology)
    for profile in calibration_profiles:
        development_tokens = read_sentences(profile.gold_treebank.development_path)
        development_corpus = encode_norwegian_sentences(
            development_tokens,
            schema=tagger.schema,
        )
        sentence_batches = tuple(
            development_corpus.sentences[start : start + tagger.batch_size]
            for start in range(
                0,
                len(development_corpus.sentences),
                tagger.batch_size,
            )
        )
        print(
            f"Collecting {profile.language_tag} development logits:",
            f"{len(sentence_batches)} batches",
        )
        for batch in iter_supervised_token_task_batches(
            tokenizer=tagger.tokenizer,
            sentence_batches=sentence_batches,
            character_vocabulary=tagger.character_vocabulary,
            maximum_character_count=tagger.maximum_character_count,
        ):
            device_batch = batch.to(device)
            logits, _ = evaluate_supervised_token_task_step(
                model=tagger.model,
                batch=device_batch,
                morphology_schema=tagger.schema.morphology,
            )
            statistics.add(
                logits=logits,
                targets=device_batch.targets,
                morphology_logit_correction=morphology_logit_correction,
            )

    head_reports = calibrate_task_heads(statistics)
    calibration = TaskTemperatureCalibration.from_head_reports(
        checkpoint_path=str(arguments.checkpoint_path),
        checkpoint_epoch_index=tagger.epoch_index,
        treebank_release=arguments.treebank_release,
        language_tags=tuple(
            profile.language_tag for profile in calibration_profiles
        ),
        morphology_logit_correction_strength=(
            arguments.morphology_logit_correction_strength
        ),
        morphology_feature_names=tuple(
            feature.name for feature in tagger.schema.morphology.features
        ),
        head_reports=head_reports,
    )
    calibration_path = (
        arguments.checkpoint_path.parent / "calibration.json"
        if arguments.calibration_path is None
        else arguments.calibration_path
    )
    write_task_temperature_calibration(calibration, calibration_path)

    print()
    header = (
        f"{'Head':24s} {'T':>7s} {'NLL before':>11s} {'NLL after':>10s} "
        f"{'ECE before':>11s} {'ECE after':>10s}"
    )
    print(header)
    for report in head_reports:
        print(
            f"{report.head_name:24s} {report.temperature:7.3f} "
            f"{report.nll_before:11.6f} {report.nll_after:10.6f} "
            f"{report.ece_before:11.6f} {report.ece_after:10.6f}"
        )
    print()
    print("Calibration:", calibration_path)


if __name__ == "__main__":
    main()
