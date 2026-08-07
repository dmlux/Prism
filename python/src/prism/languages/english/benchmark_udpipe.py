"""Benchmark UDPipe 2 on the same gold-tokenized English UD split."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from prism.conllu import Token, read_sentences
from prism.evaluation import (
    evaluate_gold_tokenized_conllu,
    serialize_universal_dependencies_evaluation_metrics,
)
from prism.evaluation.reporting import format_scalar_metric_rows
from prism.evaluation.udpipe import UdpipeRestClient
from prism.languages.english import english_profile_for_language_tag


UDPIPE_2_17_MODELS = {
    "en": "english-ewt-ud-2.17-251125",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class UdpipeBenchmarkArguments:
    language_tag: str
    gold_path: Path
    prediction_path: Path
    analysis_path: Path
    model: str
    reuse_prediction: bool
    treebank_release: str


def parse_udpipe_benchmark_arguments(
    arguments: Sequence[str] | None = None,
) -> UdpipeBenchmarkArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate UDPipe 2 with gold tokenization using official UD metrics."
        )
    )
    parser.add_argument("--language-tag", choices=("en",), default="en")
    parser.add_argument(
        "--treebank-release",
        choices=("current", "2.17"),
        default="current",
    )
    parser.add_argument("--gold", type=Path, dest="gold_path")
    parser.add_argument("--prediction", type=Path, dest="prediction_path")
    parser.add_argument("--analysis", type=Path, dest="analysis_path")
    parser.add_argument("--model")
    parser.add_argument(
        "--reuse-prediction",
        action="store_true",
        help="Evaluate an existing prediction file without contacting the service.",
    )
    parsed = parser.parse_args(arguments)

    profile = english_profile_for_language_tag(
        parsed.language_tag,
        treebank_release=parsed.treebank_release,
    )
    run_directory = (
        Path("runs") / "udpipe-2.17-251125" / f"ud-{parsed.treebank_release}"
    )
    prediction_path = parsed.prediction_path or (
        run_directory / f"{parsed.language_tag}-development.conllu"
    )
    analysis_path = parsed.analysis_path or (
        run_directory / f"{parsed.language_tag}-development-analysis.json"
    )

    return UdpipeBenchmarkArguments(
        language_tag=parsed.language_tag,
        gold_path=parsed.gold_path or profile.gold_treebank.development_path,
        prediction_path=prediction_path,
        analysis_path=analysis_path,
        model=parsed.model or UDPIPE_2_17_MODELS[parsed.language_tag],
        reuse_prediction=parsed.reuse_prediction,
        treebank_release=parsed.treebank_release,
    )


def main() -> None:
    arguments = parse_udpipe_benchmark_arguments()
    gold_text = arguments.gold_path.read_text(encoding="utf-8")

    service_seconds: float | None = None
    if arguments.reuse_prediction:
        prediction_text = arguments.prediction_path.read_text(encoding="utf-8")
    else:
        import time

        started = time.perf_counter()
        prediction_text = UdpipeRestClient().tag_gold_tokenized_conllu(
            model=arguments.model,
            conllu=gold_text,
        )
        service_seconds = time.perf_counter() - started
        arguments.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.prediction_path.write_text(prediction_text, encoding="utf-8")

    metrics = evaluate_gold_tokenized_conllu(
        gold_sentences=read_sentences(arguments.gold_path),
        system_sentences=_read_conllu_text(
            prediction_text,
            temporary_path=arguments.prediction_path,
        ),
    )

    for row in format_scalar_metric_rows(
        metric_names=("UD UPOS F1", "UD UFeats F1", "UD Lemmas F1"),
        values=(metrics.upos.f1, metrics.ufeats.f1, metrics.lemmas.f1),
    ):
        print(row)

    if service_seconds is not None:
        # Round-trip through the remote LINDAT service, including network and
        # server queueing — an upper bound, not a local model-speed number.
        token_count = sum(
            1
            for line in gold_text.splitlines()
            if line and not line.startswith("#") and "-" not in line.split("\t", 1)[0]
        )
        print()
        print(f"Service round-trip: {service_seconds:.2f} s for {token_count} tokens")

    arguments.analysis_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.analysis_path.write_text(
        json.dumps(
            {
                "system": "UDPipe 2",
                "model": arguments.model,
                "mode": "gold-tokenization",
                "treebank_release": arguments.treebank_release,
                "gold": str(arguments.gold_path),
                "prediction": str(arguments.prediction_path),
                "metrics": (
                    serialize_universal_dependencies_evaluation_metrics(metrics)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print("Analysis:", arguments.analysis_path)


def _read_conllu_text(
    text: str,
    *,
    temporary_path: Path,
) -> list[list[Token]]:
    # The prediction is persisted before parsing so one parser remains the source
    # of truth for both downloaded and reused benchmark output.
    if (
        not temporary_path.exists()
        or temporary_path.read_text(encoding="utf-8") != text
    ):
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(text, encoding="utf-8")
    return read_sentences(temporary_path)


if __name__ == "__main__":
    main()
