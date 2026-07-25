import math
from pathlib import Path

import pytest
import torch

from prism.training import (
    HeadCalibrationReport,
    TaskTemperatureCalibration,
    load_task_temperature_calibration,
    write_task_temperature_calibration,
)
from prism.training.calibration import (
    binary_expected_calibration_error,
    binary_negative_log_likelihood,
    categorical_expected_calibration_error,
    categorical_negative_log_likelihood,
    fit_binary_temperature,
    fit_categorical_temperature,
)


def _overconfident_categorical_data(
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    calibrated_logits = torch.randn((4096, 5), generator=generator)
    target_ids = torch.multinomial(
        torch.softmax(calibrated_logits, dim=-1),
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    return calibrated_logits * scale, target_ids


def test_categorical_temperature_recovers_overconfidence_scale() -> None:
    scale = 3.0
    logits, target_ids = _overconfident_categorical_data(scale)

    temperature = fit_categorical_temperature(logits=logits, target_ids=target_ids)

    assert temperature == pytest.approx(scale, rel=0.15)
    nll_before = categorical_negative_log_likelihood(
        logits=logits, target_ids=target_ids
    )
    nll_after = categorical_negative_log_likelihood(
        logits=logits, target_ids=target_ids, temperature=temperature
    )
    assert nll_after < nll_before
    ece_before = categorical_expected_calibration_error(
        logits=logits, target_ids=target_ids
    )
    ece_after = categorical_expected_calibration_error(
        logits=logits, target_ids=target_ids, temperature=temperature
    )
    assert ece_after < ece_before


def test_binary_temperature_recovers_overconfidence_scale() -> None:
    generator = torch.Generator().manual_seed(7)
    calibrated_logits = torch.randn((8192, 3), generator=generator) * 1.5
    targets = torch.bernoulli(
        torch.sigmoid(calibrated_logits),
        generator=generator,
    ).to(torch.bool)
    logits = calibrated_logits * 4.0

    temperature = fit_binary_temperature(logits=logits, targets=targets)

    assert temperature == pytest.approx(4.0, rel=0.2)
    assert binary_negative_log_likelihood(
        logits=logits, targets=targets, temperature=temperature
    ) < binary_negative_log_likelihood(logits=logits, targets=targets)
    assert binary_expected_calibration_error(
        logits=logits, targets=targets, temperature=temperature
    ) < binary_expected_calibration_error(logits=logits, targets=targets)


def test_calibrated_data_keeps_temperature_near_one() -> None:
    logits, target_ids = _overconfident_categorical_data(1.0)

    temperature = fit_categorical_temperature(logits=logits, target_ids=target_ids)

    assert temperature == pytest.approx(1.0, abs=0.15)


def test_temperature_never_changes_argmax_decisions() -> None:
    logits, _ = _overconfident_categorical_data(3.0)

    for temperature in (0.25, 1.0, 4.0):
        assert torch.equal(
            (logits / temperature).argmax(dim=-1),
            logits.argmax(dim=-1),
        )


def _report(head_name: str, temperature: float) -> HeadCalibrationReport:
    return HeadCalibrationReport(
        head_name=head_name,
        temperature=temperature,
        nll_before=0.5,
        nll_after=0.4,
        ece_before=0.08,
        ece_after=0.02,
    )


def test_calibration_artifact_round_trip(tmp_path: Path) -> None:
    calibration = TaskTemperatureCalibration.from_head_reports(
        checkpoint_path="runs/example/best.pt",
        checkpoint_epoch_index=11,
        treebank_release="current",
        language_tags=("nb", "nn"),
        morphology_logit_correction_strength=1.0,
        morphology_feature_names=("Gender", "Number"),
        head_reports=(
            _report("upos", 2.1),
            _report("lemma-rule", 1.7),
            _report("morphology:Gender", 3.2),
            _report("morphology:Number", 2.8),
        ),
    )
    path = tmp_path / "calibration.json"

    write_task_temperature_calibration(calibration, path)
    loaded = load_task_temperature_calibration(path)

    assert loaded == calibration
    assert loaded.upos_temperature == 2.1
    assert loaded.morphology_temperatures == (3.2, 2.8)


def test_calibration_artifact_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TaskTemperatureCalibration.from_head_reports(
            checkpoint_path="runs/example/best.pt",
            checkpoint_epoch_index=0,
            treebank_release="current",
            language_tags=("nb",),
            morphology_logit_correction_strength=1.0,
            morphology_feature_names=("Gender",),
            head_reports=(
                _report("upos", 1.0),
                _report("lemma-rule", 1.0),
                HeadCalibrationReport(
                    head_name="morphology:Gender",
                    temperature=math.inf,
                    nll_before=0.5,
                    nll_after=0.4,
                    ece_before=0.08,
                    ece_after=0.02,
                ),
            ),
        )
