"""Measure Prism end-to-end tagging speed on a gold-tokenized UD split.

The timed unit is one complete decision: subword tokenization, character
batching, device transfer, backbone forward, and label decoding with the
production morphology-logit correction. That mirrors what a host
application pays per request, so the reported latencies and throughputs are deployment numbers,
not bare forward-pass numbers.
"""

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from prism.conllu import read_sentences
from prism.data import encode_norwegian_sentences
from prism.languages.norwegian import norwegian_profile_for_language_tag
from prism.languages.norwegian.checkpoint_loading import (
    load_norwegian_token_tagger,
)
from prism.modeling.decoding import (
    apply_morphology_logit_correction,
    decode_token_task_logits,
)
from prism.training import (
    iter_supervised_token_task_batches,
    morphology_logit_correction_from_checkpoint,
)


_DEFAULT_BATCH_SIZES = (1, 32)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeedBenchmarkArguments:
    checkpoint_path: Path
    analysis_path: Path
    language_tag: str
    treebank_release: str
    evaluation_split: str
    device: str
    batch_sizes: tuple[int, ...]
    warmup_batch_count: int
    morphology_logit_correction_strength: float


def parse_speed_benchmark_arguments(
    arguments: Sequence[str] | None = None,
) -> SpeedBenchmarkArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Measure end-to-end Prism tagging speed on a gold-tokenized split."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        dest="checkpoint_path",
    )
    parser.add_argument("--analysis", type=Path, dest="analysis_path")
    parser.add_argument("--language-tag", choices=("nb", "nn"), default="nb")
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parser.add_argument(
        "--split",
        choices=("development", "test"),
        default="test",
        dest="evaluation_split",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        action="append",
        dest="batch_sizes",
        help=(
            "Sentences per timed request; repeat for several configurations "
            "(default: 1 and 32)."
        ),
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=8,
        dest="warmup_batch_count",
    )
    parser.add_argument(
        "--morphology-logit-correction-strength",
        type=float,
        default=0.25,
        help="Production decoding policy applied inside the timed loop.",
    )
    parsed = parser.parse_args(arguments)

    batch_sizes = tuple(parsed.batch_sizes or _DEFAULT_BATCH_SIZES)
    if any(batch_size <= 0 for batch_size in batch_sizes):
        parser.error("--batch-size values must be positive")
    if parsed.warmup_batch_count < 0:
        parser.error("--warmup-batches must not be negative")
    if not 0.0 <= parsed.morphology_logit_correction_strength <= 1.0:
        parser.error("--morphology-logit-correction-strength must be between 0 and 1")

    analysis_path = parsed.analysis_path or (
        parsed.checkpoint_path.parent
        / f"{parsed.language_tag}-{parsed.evaluation_split}-speed-{parsed.device}.json"
    )
    return SpeedBenchmarkArguments(
        checkpoint_path=parsed.checkpoint_path,
        analysis_path=analysis_path,
        language_tag=parsed.language_tag,
        treebank_release=parsed.treebank_release,
        evaluation_split=parsed.evaluation_split,
        device=parsed.device,
        batch_sizes=batch_sizes,
        warmup_batch_count=parsed.warmup_batch_count,
        morphology_logit_correction_strength=(
            parsed.morphology_logit_correction_strength
        ),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    arguments = parse_speed_benchmark_arguments()
    device = torch.device(arguments.device)

    profile = norwegian_profile_for_language_tag(
        arguments.language_tag,
        treebank_release=arguments.treebank_release,
    )
    if arguments.evaluation_split == "development":
        gold_path = profile.gold_treebank.development_path
    else:
        gold_path = profile.gold_treebank.test_path
        if gold_path is None:
            raise ValueError("The selected treebank does not expose a test split.")

    tagger = load_norwegian_token_tagger(
        checkpoint_path=arguments.checkpoint_path,
        required_language_tags=(arguments.language_tag,),
        treebank_release=arguments.treebank_release,
    )
    correction = morphology_logit_correction_from_checkpoint(
        tagger.checkpoint,
        strength=arguments.morphology_logit_correction_strength,
    )
    model = tagger.model
    model.to(device)
    model.eval()

    gold_tokens = read_sentences(gold_path)
    corpus = encode_norwegian_sentences(gold_tokens, schema=tagger.schema)
    token_count = sum(len(sentence) for sentence in gold_tokens)

    print("Checkpoint:", arguments.checkpoint_path)
    print("Split:", arguments.evaluation_split, f"({gold_path})")
    print("Device:", arguments.device)
    print("Sentences:", len(corpus.sentences))
    print("Tokens:", token_count)

    configurations = []
    for batch_size in arguments.batch_sizes:
        sentence_batches = tuple(
            corpus.sentences[start : start + batch_size]
            for start in range(0, len(corpus.sentences), batch_size)
        )

        def run_batches(batches: Sequence) -> list[float]:
            latencies: list[float] = []
            with torch.inference_mode():
                for sentence_batch in batches:
                    _synchronize(device)
                    started = time.perf_counter()
                    batch = next(
                        iter_supervised_token_task_batches(
                            tokenizer=tagger.tokenizer,
                            sentence_batches=(sentence_batch,),
                            character_vocabulary=tagger.character_vocabulary,
                            maximum_character_count=tagger.maximum_character_count,
                        )
                    ).to(device)
                    if getattr(model, "character_encoder", None) is None:
                        logits = model(batch.model_inputs)
                    else:
                        logits = model(batch.model_inputs, batch.character_inputs)
                    if correction is not None:
                        logits = apply_morphology_logit_correction(
                            logits=logits,
                            morphology_schema=tagger.schema.morphology,
                            correction=correction,
                        )
                    decode_token_task_logits(
                        logits=logits,
                        token_mask=batch.targets.token_mask,
                        morphology_schema=tagger.schema.morphology,
                    )
                    _synchronize(device)
                    latencies.append(time.perf_counter() - started)
            return latencies

        run_batches(sentence_batches[: arguments.warmup_batch_count])
        latencies = run_batches(sentence_batches)

        total_seconds = sum(latencies)
        quantiles = (
            statistics.quantiles(latencies, n=100)
            if len(latencies) >= 2
            else [latencies[0]] * 99
        )
        configuration = {
            "batch_size": batch_size,
            "request_count": len(latencies),
            "total_seconds": total_seconds,
            "sentences_per_second": len(corpus.sentences) / total_seconds,
            "tokens_per_second": token_count / total_seconds,
            "latency_mean_milliseconds": 1000.0 * total_seconds / len(latencies),
            "latency_p50_milliseconds": 1000.0 * quantiles[49],
            "latency_p95_milliseconds": 1000.0 * quantiles[94],
        }
        configurations.append(configuration)
        print()
        print(f"Batch size {batch_size}:")
        print(f"  Total wall-clock      {total_seconds:.2f} s")
        print(f"  Sentences per second  {configuration['sentences_per_second']:.1f}")
        print(f"  Tokens per second     {configuration['tokens_per_second']:.0f}")
        print(
            "  Latency mean/p50/p95  "
            f"{configuration['latency_mean_milliseconds']:.1f} / "
            f"{configuration['latency_p50_milliseconds']:.1f} / "
            f"{configuration['latency_p95_milliseconds']:.1f} ms"
        )

    arguments.analysis_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.analysis_path.write_text(
        json.dumps(
            {
                "checkpoint": str(arguments.checkpoint_path),
                "language_tag": arguments.language_tag,
                "evaluation_split": arguments.evaluation_split,
                "device": arguments.device,
                "sentence_count": len(corpus.sentences),
                "token_count": token_count,
                "warmup_batch_count": arguments.warmup_batch_count,
                "morphology_logit_correction_strength": (
                    arguments.morphology_logit_correction_strength
                ),
                "timed_scope": (
                    "tokenization + character batching + device transfer "
                    "+ forward + logit correction + decoding"
                ),
                "configurations": configurations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print("Analysis:", arguments.analysis_path)


if __name__ == "__main__":
    main()
