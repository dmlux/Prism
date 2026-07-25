"""Post-hoc per-task-head temperature calibration.

Neural probabilities are frequently overconfident, and the discretely
selected teacher/labeler checkpoints are deliberately chosen for their
decisions rather than their negative log-likelihood. Temperature scaling
repairs that overconfidence after training: every task head receives one
scalar temperature that divides its logits before the probability function.
The argmax decisions never change, only the confidence values do — which is
exactly what confidence-filtered silver labeling requires.

The fit is a deterministic two-stage grid search over the log-temperature
that minimizes the head's development negative log-likelihood. Categorical
heads (UPOS, lemma rules, exclusive morphology features) use Cross-Entropy;
multi-valued morphology features use Binary Cross-Entropy over their real
value logits, matching the training objective exactly.
"""

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

from prism.data.batches import TokenTaskTargetBatch
from prism.modeling.decoding import (
    MorphologyLogitCorrection,
    apply_morphology_logit_correction,
)
from prism.modeling.outputs import TokenTaskLogits
from prism.schema import MorphologySchema


TASK_TEMPERATURE_CALIBRATION_FORMAT_VERSION = 1

_COARSE_LOG_TEMPERATURES = tuple(
    -1.5 + 3.0 * index / 30 for index in range(31)
)
_REFINEMENT_STEP_COUNT = 40
_ECE_BIN_COUNT = 15


def _grid_fit_temperature(nll_of_temperature) -> float:
    """Two-stage deterministic grid search over the log10 temperature."""

    def best_log_temperature(candidates: Sequence[float]) -> float:
        scores = [
            (nll_of_temperature(10.0**candidate), candidate)
            for candidate in candidates
        ]
        return min(scores)[1]

    coarse = best_log_temperature(_COARSE_LOG_TEMPERATURES)
    step = _COARSE_LOG_TEMPERATURES[1] - _COARSE_LOG_TEMPERATURES[0]
    fine_candidates = tuple(
        coarse - step + 2.0 * step * index / _REFINEMENT_STEP_COUNT
        for index in range(_REFINEMENT_STEP_COUNT + 1)
    )
    return 10.0 ** best_log_temperature(fine_candidates)


def categorical_negative_log_likelihood(
    *,
    logits: Tensor,
    target_ids: Tensor,
    temperature: float = 1.0,
) -> float:
    return float(
        functional.cross_entropy(logits / temperature, target_ids).item()
    )


def binary_negative_log_likelihood(
    *,
    logits: Tensor,
    targets: Tensor,
    temperature: float = 1.0,
) -> float:
    return float(
        functional.binary_cross_entropy_with_logits(
            logits / temperature,
            targets.to(logits.dtype),
        ).item()
    )


def fit_categorical_temperature(*, logits: Tensor, target_ids: Tensor) -> float:
    if logits.ndim != 2 or target_ids.ndim != 1:
        raise ValueError("Categorical calibration expects [N, C] logits and [N] IDs.")
    if logits.shape[0] != target_ids.shape[0] or logits.shape[0] == 0:
        raise ValueError("Categorical calibration requires matching, non-empty data.")
    return _grid_fit_temperature(
        lambda temperature: categorical_negative_log_likelihood(
            logits=logits,
            target_ids=target_ids,
            temperature=temperature,
        )
    )


def fit_binary_temperature(*, logits: Tensor, targets: Tensor) -> float:
    if logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError("Binary calibration requires matching, non-empty data.")
    return _grid_fit_temperature(
        lambda temperature: binary_negative_log_likelihood(
            logits=logits,
            targets=targets,
            temperature=temperature,
        )
    )


def categorical_expected_calibration_error(
    *,
    logits: Tensor,
    target_ids: Tensor,
    temperature: float = 1.0,
) -> float:
    probabilities = functional.softmax(logits / temperature, dim=-1)
    confidences, predictions = probabilities.max(dim=-1)
    correct = predictions == target_ids
    return _expected_calibration_error(confidences=confidences, correct=correct)


def binary_expected_calibration_error(
    *,
    logits: Tensor,
    targets: Tensor,
    temperature: float = 1.0,
) -> float:
    probabilities = torch.sigmoid(logits / temperature).reshape(-1)
    flat_targets = targets.reshape(-1).to(torch.bool)
    predictions = probabilities >= 0.5
    confidences = torch.where(predictions, probabilities, 1.0 - probabilities)
    return _expected_calibration_error(
        confidences=confidences,
        correct=predictions == flat_targets,
    )


def _expected_calibration_error(*, confidences: Tensor, correct: Tensor) -> float:
    total = confidences.shape[0]
    if total == 0:
        raise ValueError("Expected calibration error requires predictions.")
    error = 0.0
    for bin_index in range(_ECE_BIN_COUNT):
        lower = bin_index / _ECE_BIN_COUNT
        upper = (bin_index + 1) / _ECE_BIN_COUNT
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_count = int(in_bin.sum().item())
        if bin_count == 0:
            continue
        bin_confidence = float(confidences[in_bin].mean().item())
        bin_accuracy = float(correct[in_bin].to(torch.float32).mean().item())
        error += (bin_count / total) * abs(bin_confidence - bin_accuracy)
    return error


@dataclass(slots=True, kw_only=True)
class _FeatureStatistics:
    logit_batches: list[Tensor]
    target_batches: list[Tensor]


@dataclass(slots=True, kw_only=True)
class CalibrationStatistics:
    """Masked development logits and targets collected per task head."""

    morphology_schema: MorphologySchema
    upos_logit_batches: list[Tensor]
    upos_target_batches: list[Tensor]
    lemma_logit_batches: list[Tensor]
    lemma_target_batches: list[Tensor]
    feature_statistics: tuple[_FeatureStatistics, ...]

    @classmethod
    def empty(cls, morphology_schema: MorphologySchema) -> "CalibrationStatistics":
        return cls(
            morphology_schema=morphology_schema,
            upos_logit_batches=[],
            upos_target_batches=[],
            lemma_logit_batches=[],
            lemma_target_batches=[],
            feature_statistics=tuple(
                _FeatureStatistics(logit_batches=[], target_batches=[])
                for _ in morphology_schema.features
            ),
        )

    def add(
        self,
        *,
        logits: TokenTaskLogits,
        targets: TokenTaskTargetBatch,
        morphology_logit_correction: MorphologyLogitCorrection | None,
    ) -> None:
        corrected = (
            logits
            if morphology_logit_correction is None
            else apply_morphology_logit_correction(
                logits=logits,
                morphology_schema=self.morphology_schema,
                correction=morphology_logit_correction,
            )
        )
        token_mask = targets.token_mask
        lemma_mask = token_mask & targets.lemma_rule_mask

        self.upos_logit_batches.append(
            corrected.upos_logits[token_mask].detach().cpu()
        )
        self.upos_target_batches.append(
            targets.upos_ids[token_mask].detach().cpu()
        )
        self.lemma_logit_batches.append(
            corrected.lemma_rule_logits[lemma_mask].detach().cpu()
        )
        self.lemma_target_batches.append(
            targets.lemma_rule_ids[lemma_mask].detach().cpu()
        )
        for statistics, feature_logits, feature_targets, feature in zip(
            self.feature_statistics,
            corrected.morphology_logits,
            targets.morphology_targets,
            self.morphology_schema.features,
            strict=True,
        ):
            masked_logits = feature_logits[token_mask].detach().cpu()
            masked_targets = feature_targets[token_mask].detach().cpu()
            statistics.logit_batches.append(masked_logits)
            if feature.allows_multiple_values:
                statistics.target_batches.append(masked_targets[..., 1:])
            else:
                statistics.target_batches.append(
                    masked_targets.to(torch.long).argmax(dim=-1)
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class HeadCalibrationReport:
    head_name: str
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float

    def __post_init__(self) -> None:
        if not self.head_name or self.head_name.strip() != self.head_name:
            raise ValueError("Calibration head name must be non-empty and trimmed.")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("Calibration temperature must be finite and positive.")


def calibrate_task_heads(
    statistics: CalibrationStatistics,
) -> tuple[HeadCalibrationReport, ...]:
    """Fit one temperature per task head and report NLL/ECE before and after."""

    reports: list[HeadCalibrationReport] = []

    def categorical_report(
        head_name: str,
        logit_batches: Iterable[Tensor],
        target_batches: Iterable[Tensor],
    ) -> HeadCalibrationReport:
        logits = torch.cat(tuple(logit_batches))
        target_ids = torch.cat(tuple(target_batches))
        temperature = fit_categorical_temperature(
            logits=logits,
            target_ids=target_ids,
        )
        return HeadCalibrationReport(
            head_name=head_name,
            temperature=temperature,
            nll_before=categorical_negative_log_likelihood(
                logits=logits, target_ids=target_ids
            ),
            nll_after=categorical_negative_log_likelihood(
                logits=logits, target_ids=target_ids, temperature=temperature
            ),
            ece_before=categorical_expected_calibration_error(
                logits=logits, target_ids=target_ids
            ),
            ece_after=categorical_expected_calibration_error(
                logits=logits, target_ids=target_ids, temperature=temperature
            ),
        )

    reports.append(
        categorical_report(
            "upos",
            statistics.upos_logit_batches,
            statistics.upos_target_batches,
        )
    )
    reports.append(
        categorical_report(
            "lemma-rule",
            statistics.lemma_logit_batches,
            statistics.lemma_target_batches,
        )
    )
    for statistics_for_feature, feature in zip(
        statistics.feature_statistics,
        statistics.morphology_schema.features,
        strict=True,
    ):
        if feature.allows_multiple_values:
            logits = torch.cat(tuple(statistics_for_feature.logit_batches))
            targets = torch.cat(tuple(statistics_for_feature.target_batches))
            temperature = fit_binary_temperature(logits=logits, targets=targets)
            reports.append(
                HeadCalibrationReport(
                    head_name=f"morphology:{feature.name}",
                    temperature=temperature,
                    nll_before=binary_negative_log_likelihood(
                        logits=logits, targets=targets
                    ),
                    nll_after=binary_negative_log_likelihood(
                        logits=logits, targets=targets, temperature=temperature
                    ),
                    ece_before=binary_expected_calibration_error(
                        logits=logits, targets=targets
                    ),
                    ece_after=binary_expected_calibration_error(
                        logits=logits, targets=targets, temperature=temperature
                    ),
                )
            )
        else:
            reports.append(
                categorical_report(
                    f"morphology:{feature.name}",
                    statistics_for_feature.logit_batches,
                    statistics_for_feature.target_batches,
                )
            )
    return tuple(reports)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskTemperatureCalibration:
    """Versioned, provenance-carrying per-head temperature artifact."""

    format_version: int
    checkpoint_path: str
    checkpoint_epoch_index: int
    treebank_release: str
    language_tags: tuple[str, ...]
    morphology_logit_correction_strength: float
    upos_temperature: float
    lemma_rule_temperature: float
    morphology_feature_names: tuple[str, ...]
    morphology_temperatures: tuple[float, ...]
    head_reports: tuple[HeadCalibrationReport, ...]

    def __post_init__(self) -> None:
        if self.format_version != TASK_TEMPERATURE_CALIBRATION_FORMAT_VERSION:
            raise ValueError("Unsupported calibration format version.")
        if not self.checkpoint_path:
            raise ValueError("Calibration must reference its checkpoint.")
        if self.checkpoint_epoch_index < 0:
            raise ValueError("Calibration epoch index must not be negative.")
        if not self.language_tags:
            raise ValueError("Calibration must record its language tags.")
        if len(self.morphology_feature_names) != len(self.morphology_temperatures):
            raise ValueError(
                "Calibration feature names and temperatures must match."
            )
        if not self.morphology_feature_names:
            raise ValueError("Calibration must cover the morphology features.")
        for temperature in (
            self.upos_temperature,
            self.lemma_rule_temperature,
            *self.morphology_temperatures,
        ):
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(
                    "Calibration temperatures must be finite and positive."
                )

    @classmethod
    def from_head_reports(
        cls,
        *,
        checkpoint_path: str,
        checkpoint_epoch_index: int,
        treebank_release: str,
        language_tags: Sequence[str],
        morphology_logit_correction_strength: float,
        morphology_feature_names: Sequence[str],
        head_reports: Sequence[HeadCalibrationReport],
    ) -> "TaskTemperatureCalibration":
        temperatures = {report.head_name: report.temperature for report in head_reports}
        return cls(
            format_version=TASK_TEMPERATURE_CALIBRATION_FORMAT_VERSION,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch_index=checkpoint_epoch_index,
            treebank_release=treebank_release,
            language_tags=tuple(language_tags),
            morphology_logit_correction_strength=(
                morphology_logit_correction_strength
            ),
            upos_temperature=temperatures["upos"],
            lemma_rule_temperature=temperatures["lemma-rule"],
            morphology_feature_names=tuple(morphology_feature_names),
            morphology_temperatures=tuple(
                temperatures[f"morphology:{name}"]
                for name in morphology_feature_names
            ),
            head_reports=tuple(head_reports),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibratedTaskProbabilities:
    """Temperature-scaled probabilities of one already-corrected batch."""

    upos_probabilities: Tensor
    morphology_probabilities: tuple[Tensor, ...]
    lemma_rule_probabilities: Tensor


def calibrated_task_probabilities(
    *,
    logits: TokenTaskLogits,
    morphology_schema: MorphologySchema,
    calibration: TaskTemperatureCalibration,
) -> CalibratedTaskProbabilities:
    """Turn corrected task logits into calibrated probabilities.

    Exclusive morphology features produce a softmax over their complete label
    space including ``<NONE>``; multi-valued features produce independent
    sigmoid probabilities over their real value logits, matching the training
    objective and the stored temperature semantics.
    """

    feature_names = tuple(
        feature.name for feature in morphology_schema.features
    )
    if feature_names != calibration.morphology_feature_names:
        raise ValueError(
            "Calibration morphology features do not match the schema."
        )

    morphology_probabilities: list[Tensor] = []
    for feature, feature_logits, temperature in zip(
        morphology_schema.features,
        logits.morphology_logits,
        calibration.morphology_temperatures,
        strict=True,
    ):
        scaled = feature_logits / temperature
        if feature.allows_multiple_values:
            morphology_probabilities.append(torch.sigmoid(scaled))
        else:
            morphology_probabilities.append(torch.softmax(scaled, dim=-1))

    return CalibratedTaskProbabilities(
        upos_probabilities=torch.softmax(
            logits.upos_logits / calibration.upos_temperature,
            dim=-1,
        ),
        morphology_probabilities=tuple(morphology_probabilities),
        lemma_rule_probabilities=torch.softmax(
            logits.lemma_rule_logits / calibration.lemma_rule_temperature,
            dim=-1,
        ),
    )


def write_task_temperature_calibration(
    calibration: TaskTemperatureCalibration,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(calibration), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_task_temperature_calibration(path: Path) -> TaskTemperatureCalibration:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Calibration artifact must contain a JSON object.")
    head_reports = value.pop("head_reports", None)
    if not isinstance(head_reports, list):
        raise ValueError("Calibration artifact must contain head reports.")
    return TaskTemperatureCalibration(
        **{
            **value,
            "language_tags": tuple(value["language_tags"]),
            "morphology_feature_names": tuple(value["morphology_feature_names"]),
            "morphology_temperatures": tuple(value["morphology_temperatures"]),
        },
        head_reports=tuple(
            HeadCalibrationReport(**report) for report in head_reports
        ),
    )
