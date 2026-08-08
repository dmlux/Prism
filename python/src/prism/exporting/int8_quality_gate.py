"""Reproducible int8 quality gate: fp32 export adapter versus int8 twin.

The shipped int8 artifact must decode like its fp32 reference. This module
measures that directly and reproducibly, language-independently: it builds the
int8 eager twin through the *same* committed
:class:`~prism.exporting.quantization.Int8QuantizationStrategy` the export uses
(so what is measured is exactly what ships), runs the fp32 adapter and the
int8 twin over a development-split batch stream with the identical production
decoding policy, and scores both against gold. It reports per-task fp32/int8
accuracy, the int8−fp32 delta, and the decision-flip rate.

The language bindings supply the pieces through :class:`DevelopmentBatch` and a
``decodable`` callable (their ``_decodable_flat_outputs``); everything else —
quantization, decoding, scoring — lives here so both languages gate identically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.conllu import Token
from prism.evaluation.universal_dependencies import evaluate_gold_tokenized_conllu
from prism.exporting.artifact import (
    decoded_sentence_predictions,
    token_task_logits_from_flat_outputs,
)
from prism.exporting.quantization import Int8QuantizationStrategy

# outputs -> decode-equivalent flat logits (a binding's _decodable_flat_outputs).
DecodableOutputs = Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]]
# (token_form, normalized_lemma, predicted_upos) -> lemma as scored against gold.
# The language's UD lemma decoder; it restores any normalization marker (the
# Norwegian ``$`` punctuation marker) so decoded lemmas match the raw treebank.
LemmaDecoder = Callable[[str, str, str], str]

TASKS = ("upos", "feats", "lemma")


@dataclass(frozen=True)
class DevelopmentBatch:
    """One fixed-shape batch of development sentences and its gold labels."""

    input_tensors: tuple[Tensor, ...]
    token_mask: Tensor
    # Per-sentence model-input tokens, used to reconstruct decoded predictions.
    sentence_tokens: tuple[Sequence[object], ...]
    # Per-sentence gold tokens, each exposing ``upos``, ``features``, ``lemma``.
    gold_tokens: tuple[Sequence[object], ...]


@dataclass(frozen=True)
class TaskQualityDelta:
    task: str
    fp32_accuracy: float
    int8_accuracy: float
    decision_flips: int

    @property
    def delta(self) -> float:
        return self.int8_accuracy - self.fp32_accuracy


@dataclass(frozen=True)
class Int8QualityReport:
    language_tag: str
    token_count: int
    tasks: tuple[TaskQualityDelta, ...]

    @property
    def worst_regression(self) -> float:
        """Most negative int8−fp32 delta (0.0 if int8 never regresses)."""

        return min(0.0, *(task.delta for task in self.tasks))

    def format_table(self) -> str:
        lines = [f"{self.language_tag} development ({self.token_count} tokens):"]
        for task in self.tasks:
            flip_rate = 100.0 * task.decision_flips / self.token_count
            lines.append(
                f"  {task.task:6s} fp32 {task.fp32_accuracy:.4f}%  "
                f"int8 {task.int8_accuracy:.4f}%  delta {task.delta:+.4f} pp  "
                f"flips {task.decision_flips} ({flip_rate:.3f}%)"
            )
        return "\n".join(lines)


def build_int8_twin(
    *,
    adapter: nn.Module,
    strategy: Int8QuantizationStrategy,
    calibration_batches: Sequence[tuple[Tensor, ...]],
) -> nn.Module:
    """Fold (if needed) and quantize the adapter through the shipped strategy."""

    if not strategy.supports_int8():
        raise ValueError("The selected quantization strategy does not support int8.")
    prepared = strategy.prepare_float_adapter(adapter)
    return strategy.quantize(
        adapter=prepared,
        calibration_batches=calibration_batches,
    )


def _system_token(decoded_token: dict, lemma_decoder: LemmaDecoder) -> Token:
    """A decoded prediction as a UD ``Token`` for the official scorer.

    Decoded morphology is ``name -> [values]`` (empty when inactive); the UD
    scorer compares ``name=value`` bundles, so keep only active features and
    join multi-values the UD way (comma-separated, sorted). The lemma is passed
    through the language's UD lemma decoder to restore any normalization marker
    (so it is scored against the raw treebank gold exactly as the frozen
    benchmark does).
    """

    return Token(
        text=decoded_token["form"],
        lemma=lemma_decoder(
            decoded_token["form"], decoded_token["lemma"], decoded_token["upos"]
        ),
        upos=decoded_token["upos"],
        features={
            name: ",".join(sorted(values))
            for name, values in decoded_token["morphology"].items()
            if values
        },
        space_after=True,
    )


def evaluate_int8_quality(
    *,
    language_tag: str,
    schema: object,
    fp32_adapter: nn.Module,
    int8_twin: nn.Module,
    batches: Iterable[DevelopmentBatch],
    decodable: DecodableOutputs,
    lemma_decoder: LemmaDecoder,
) -> Int8QualityReport:
    """Score the fp32 adapter and int8 twin against gold over the batches.

    Both variants decode with the production policy; scoring uses the official
    gold-tokenized UD evaluator (:func:`evaluate_gold_tokenized_conllu`), so the
    absolute numbers match the frozen benchmark methodology exactly and only the
    int8/fp32 difference is attributable to quantization.
    """

    def decode(outputs: tuple[Tensor, ...], batch: DevelopmentBatch):
        return decoded_sentence_predictions(
            schema=schema,
            logits=token_task_logits_from_flat_outputs(
                decodable(outputs),
                schema=schema,
            ),
            token_mask=batch.token_mask,
            sentence_tokens=batch.sentence_tokens,
        )

    gold_sentences: list[tuple[Token, ...]] = []
    fp32_sentences: list[tuple[Token, ...]] = []
    int8_sentences: list[tuple[Token, ...]] = []
    flips = {task: 0 for task in TASKS}
    token_count = 0

    for batch in batches:
        with torch.no_grad():
            fp32_outputs = fp32_adapter(*batch.input_tensors)
            int8_outputs = int8_twin(*batch.input_tensors)
        fp32_decoded = decode(fp32_outputs, batch)
        int8_decoded = decode(int8_outputs, batch)

        for gold_tokens, fp32_sentence, int8_sentence in zip(
            batch.gold_tokens, fp32_decoded, int8_decoded, strict=True
        ):
            gold_sentences.append(tuple(gold_tokens))
            fp32_sentences.append(
                tuple(
                    _system_token(token, lemma_decoder)
                    for token in fp32_sentence["tokens"]
                )
            )
            int8_sentences.append(
                tuple(
                    _system_token(token, lemma_decoder)
                    for token in int8_sentence["tokens"]
                )
            )
            for fp32_token, int8_token in zip(
                fp32_sentence["tokens"], int8_sentence["tokens"], strict=True
            ):
                token_count += 1
                flips["upos"] += fp32_token["upos"] != int8_token["upos"]
                flips["feats"] += fp32_token["morphology"] != int8_token["morphology"]
                flips["lemma"] += fp32_token["lemma"] != int8_token["lemma"]

    if token_count == 0:
        raise ValueError("The development batch stream produced no tokens to score.")

    fp32_metrics = evaluate_gold_tokenized_conllu(
        gold_sentences=gold_sentences, system_sentences=fp32_sentences
    )
    int8_metrics = evaluate_gold_tokenized_conllu(
        gold_sentences=gold_sentences, system_sentences=int8_sentences
    )
    scores = {
        "upos": (fp32_metrics.upos, int8_metrics.upos),
        "feats": (fp32_metrics.ufeats, int8_metrics.ufeats),
        "lemma": (fp32_metrics.lemmas, int8_metrics.lemmas),
    }
    tasks = tuple(
        TaskQualityDelta(
            task=task,
            fp32_accuracy=100.0 * scores[task][0].f1,
            int8_accuracy=100.0 * scores[task][1].f1,
            decision_flips=flips[task],
        )
        for task in TASKS
    )
    return Int8QualityReport(
        language_tag=language_tag,
        token_count=token_count,
        tasks=tasks,
    )
