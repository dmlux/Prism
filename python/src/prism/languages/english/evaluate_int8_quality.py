"""int8 quality gate for the English (Ettin/ModernBERT) model.

Runs the fp32 export adapter and its int8 twin over the English development
split with the identical production decoding policy and reports per-task
accuracy, the int8-fp32 delta, and the decision-flip rate. The shared,
language-independent engine lives in
:mod:`prism.exporting.int8_quality_gate`; this module only wires the English
loaders. Norwegian has the mirror module.

    python -m prism.languages.english.evaluate_int8_quality \\
        --checkpoint runs/en-student-distilled/best-development-task-accuracy.pt \\
        --calibration runs/en-student-distilled/calibration.json \\
        --treebank-release 2.17
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import torch

from prism.conllu import read_sentences
from prism.data import encode_english_sentences
from prism.data.english import build_english_ud_lemma_decoder
from prism.exporting import FixedExportShapes
from prism.exporting.int8_quality_gate import (
    DevelopmentBatch,
    build_int8_twin,
    evaluate_int8_quality,
)
from prism.exporting.quantization import resolve_int8_quantization_strategy
from prism.languages.english.checkpoint_loading import load_english_token_tagger
from prism.languages.english.export_artifact import (
    _decodable_flat_outputs,
    build_export_adapter,
    build_fixture_input_tensors,
)
from prism.languages.english.profile import english_training_profiles_for_language_tag
from prism.training.calibration import load_task_temperature_calibration


def _development_batches(
    *, profile, tagger, shapes: FixedExportShapes, limit: int | None = None
) -> Iterator[DevelopmentBatch]:
    raw_sentences = tuple(read_sentences(profile.gold_treebank.development_path))
    corpus = encode_english_sentences(raw_sentences, schema=tagger.schema)
    # read_sentences yields gold Token lists; encode_* preserves order 1:1, so
    # pair each encoded model input with its gold tokens by index.
    paired = [
        (encoded.model_input, gold_tokens)
        for gold_tokens, encoded in zip(raw_sentences, corpus.sentences, strict=True)
        if len(encoded.model_input.tokens) <= shapes.token_count
    ]
    if limit is not None:
        paired = paired[:limit]
    for start in range(0, len(paired), shapes.batch_size):
        chunk = paired[start : start + shapes.batch_size]
        model_inputs = tuple(model_input for model_input, _ in chunk)
        named_inputs = build_fixture_input_tensors(
            sentences=model_inputs, tagger=tagger, shapes=shapes
        )
        by_name = dict(named_inputs)
        yield DevelopmentBatch(
            input_tensors=tuple(tensor for _, tensor in named_inputs),
            token_mask=by_name["token_mask"],
            sentence_tokens=tuple(model_input.tokens for model_input in model_inputs),
            gold_tokens=tuple(gold_tokens for _, gold_tokens in chunk),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-split int8-vs-fp32 quality gate for prism-en."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, dest="checkpoint_path")
    parser.add_argument(
        "--calibration", type=Path, default=None, dest="calibration_path"
    )
    parser.add_argument("--treebank-release", choices=("current", "2.17"), default="2.17")
    parser.add_argument("--language-tag", choices=("en",), default="en")
    parser.add_argument(
        "--morphology-logit-correction-strength",
        type=float,
        default=None,
        help="Defaults to the calibration's fitted strength when a calibration "
        "is given, else 0.0.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--subword-count", type=int, default=160)
    parser.add_argument("--token-count", type=int, default=96)
    parser.add_argument("--character-count", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap sentences per split (smoke runs)."
    )
    parser.add_argument("--calibration-batches", type=int, default=2)
    parser.add_argument("--json", type=Path, default=None, dest="json_path")
    parser.add_argument(
        "--acceptance",
        type=float,
        default=0.05,
        help="Fail if any task regresses by more than this many pp.",
    )
    arguments = parser.parse_args()

    torch.manual_seed(0)
    profiles = english_training_profiles_for_language_tag(
        arguments.language_tag, treebank_release=arguments.treebank_release
    )
    tagger = load_english_token_tagger(
        checkpoint_path=arguments.checkpoint_path,
        required_language_tags=tuple(profile.language_tag for profile in profiles),
        treebank_release=arguments.treebank_release,
    )
    calibration = (
        None
        if arguments.calibration_path is None
        else load_task_temperature_calibration(arguments.calibration_path)
    )
    if arguments.morphology_logit_correction_strength is not None:
        strength = arguments.morphology_logit_correction_strength
    elif calibration is not None:
        strength = calibration.morphology_logit_correction_strength
    else:
        strength = 0.0

    adapter = build_export_adapter(
        tagger,
        morphology_logit_correction_strength=strength,
        calibration=calibration,
    )
    shapes = FixedExportShapes(
        batch_size=arguments.batch_size,
        subword_count=arguments.subword_count,
        token_count=arguments.token_count,
        character_count=arguments.character_count,
    )

    # Calibration batches for the quantizer: the first development batches.
    calibration_batches = [
        batch.input_tensors
        for profile in profiles
        for batch in _development_batches(
            profile=profile,
            tagger=tagger,
            shapes=shapes,
            limit=arguments.calibration_batches * arguments.batch_size,
        )
    ][: max(1, arguments.calibration_batches * len(profiles))]

    strategy = resolve_int8_quantization_strategy(profiles[0].quantization)
    # Builds the twin through the shipped strategy; folds the adapter in place
    # for NorBERT4 (a no-op for ModernBERT), so `adapter` is the fp32 reference.
    int8_twin = build_int8_twin(
        adapter=adapter,
        strategy=strategy,
        calibration_batches=calibration_batches,
    )

    # English needs no lemma-marker restoration (identity decoder), wired the
    # same way as Norwegian so the shared engine stays language-agnostic.
    lemma_decoder = build_english_ud_lemma_decoder(
        [
            sentence
            for profile in profiles
            for sentence in read_sentences(profile.gold_treebank.training_path)
        ]
    )

    reports = []
    worst = 0.0
    for profile in profiles:
        report = evaluate_int8_quality(
            language_tag=profile.language_tag,
            schema=tagger.schema,
            fp32_adapter=adapter,
            int8_twin=int8_twin,
            batches=_development_batches(
                profile=profile,
                tagger=tagger,
                shapes=shapes,
                limit=arguments.limit,
            ),
            decodable=partial(_decodable_flat_outputs, tagger=tagger, calibrated=True),
            lemma_decoder=lemma_decoder,
        )
        print("\n" + report.format_table())
        worst = min(worst, report.worst_regression)
        reports.append(report)

    print(f"\nWorst int8 regression: {worst:+.4f} pp (acceptance ±{arguments.acceptance})")
    if arguments.json_path is not None:
        arguments.json_path.write_text(
            json.dumps(
                {
                    report.language_tag: {
                        task.task: {
                            "fp32": task.fp32_accuracy,
                            "int8": task.int8_accuracy,
                            "delta": task.delta,
                            "flips": task.decision_flips,
                        }
                        for task in report.tasks
                    }
                    for report in reports
                },
                indent=2,
            )
        )
    if worst < -abs(arguments.acceptance):
        raise SystemExit(
            f"int8 quality gate failed: {worst:+.4f} pp exceeds the "
            f"±{arguments.acceptance} pp acceptance."
        )


if __name__ == "__main__":
    main()
