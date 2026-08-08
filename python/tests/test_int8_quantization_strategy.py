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


def test_modernbert_strategy_supports_int8() -> None:
    strategy = resolve_int8_quantization_strategy("modernbert")
    assert isinstance(strategy, ModernBertInt8Strategy)
    assert strategy.supports_int8()


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
    assert resolve_int8_quantization_strategy(
        ENGLISH_PROFILE.quantization
    ).supports_int8()


def test_clean_modernbert_masks_match_transformers_bit_for_bit() -> None:
    """The export-clean mask twins must equal the stock builders exactly.

    This is the parity guarantee behind int8 English export: the quantized
    graph uses the clean masks, so any drift from transformers' own masks would
    silently change the model's outputs. Guards against transformers upgrades.
    """

    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip(
        "transformers.models.modernbert.modeling_modernbert"
    )
    from transformers import ModernBertConfig

    from prism.exporting.quantization import (
        _clean_bidirectional_mask,
        _clean_sliding_window_mask,
    )

    config = ModernBertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
    )
    config._attn_implementation = "eager"
    config.sliding_window = 3

    for batch, seq in [(2, 12), (8, 24)]:
        embeds = torch.zeros(batch, seq, config.hidden_size)
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        attention_mask[0, seq - 2:] = 0
        for real_name, clean_fn in [
            ("create_bidirectional_mask", _clean_bidirectional_mask),
            ("create_bidirectional_sliding_window_mask", _clean_sliding_window_mask),
        ]:
            real = getattr(modeling, real_name)(
                config=config, inputs_embeds=embeds, attention_mask=attention_mask
            )
            clean = clean_fn(config, embeds, attention_mask)
            assert real is not None
            assert real.shape == clean.shape
            assert torch.equal(real, clean)
