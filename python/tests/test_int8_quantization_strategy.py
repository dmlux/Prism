import pytest

from prism.exporting import (
    ModernBertInt8Strategy,
    XnnpackEmbeddingDynamicInt8Strategy,
    resolve_int8_quantization_strategy,
)
from prism.exporting.quantization import DEFAULT_INT8_QUANTIZATION
from prism.languages.english import ENGLISH_PROFILE
from prism.languages.norwegian import NORWEGIAN_BOKMAAL_PROFILE


def test_default_strategy_supports_int8() -> None:
    strategy = resolve_int8_quantization_strategy("xnnpack-embedding-dynamic")
    assert isinstance(strategy, XnnpackEmbeddingDynamicInt8Strategy)
    assert strategy.supports_int8()


def test_modernbert_strategy_declines_int8() -> None:
    strategy = resolve_int8_quantization_strategy("modernbert")
    assert isinstance(strategy, ModernBertInt8Strategy)
    assert not strategy.supports_int8()
    with pytest.raises(NotImplementedError, match="ModernBERT"):
        strategy.quantize(adapter=None, calibration_batches=())


def test_modernbert_prepare_float_adapter_is_identity() -> None:
    strategy = ModernBertInt8Strategy()
    sentinel = object()
    assert strategy.prepare_float_adapter(sentinel) is sentinel


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown int8 quantization strategy"):
        resolve_int8_quantization_strategy("bogus")


def test_norwegian_profile_uses_the_default_quantization() -> None:
    assert NORWEGIAN_BOKMAAL_PROFILE.quantization == DEFAULT_INT8_QUANTIZATION
    assert resolve_int8_quantization_strategy(
        NORWEGIAN_BOKMAAL_PROFILE.quantization
    ).supports_int8()


def test_english_profile_selects_the_modernbert_quantization() -> None:
    assert ENGLISH_PROFILE.quantization == "modernbert"
    assert not resolve_int8_quantization_strategy(
        ENGLISH_PROFILE.quantization
    ).supports_int8()
