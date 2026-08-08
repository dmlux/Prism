"""Backbone-specific int8 quantization strategies for artifact export.

Quantization is architecture-specific, so it lives behind a language- and
backbone-independent interface instead of being wired into the shared export
flow. Each backbone family supplies its own strategy:

* the GPT-BERT / NorBERT4 graph quantizes cleanly with the standard XNNPACK
  dynamic-linear + per-channel-embedding PT2E path
  (:class:`XnnpackEmbeddingDynamicInt8Strategy`, the historical default);
* ModernBERT / Ettin does not yet, because its sliding-window attention mask
  indexes ``attention_mask`` with ``arange``-derived integer index tensors and
  PT2E floats one of them during the calibration forward
  (:class:`ModernBertInt8Strategy`).

The fp32 export path never touches this module, so it is unchanged for every
backbone. A profile selects its strategy by a string discriminator
(``LanguageProfileSpec.quantization``); :func:`resolve_int8_quantization_strategy`
maps that to an instance at export time, keeping ``prism.exporting`` free of any
dependency on ``prism.languages``.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


DEFAULT_INT8_QUANTIZATION = "xnnpack-embedding-dynamic"


def _require_executorch_module(module_path: str) -> object:
    import importlib

    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Model export requires the optional ExecuTorch dependency; "
            "install it with: python -m pip install -e './python[export]'"
        ) from error


@runtime_checkable
class Int8QuantizationStrategy(Protocol):
    """How one backbone family lowers to an int8 program.

    ``prepare_float_adapter`` performs any backbone-specific graph surgery the
    quantizer needs on the floating adapter (and returns the adapter that is
    also lowered as the float program); ``quantize`` returns the converted int8
    eager twin used for fixtures and lowering. ``supports_int8`` lets the export
    fail fast with a clear message on backbones that cannot be quantized yet.
    """

    def supports_int8(self) -> bool: ...

    def prepare_float_adapter(self, adapter: nn.Module) -> nn.Module: ...

    def quantize(
        self,
        *,
        adapter: nn.Module,
        calibration_batches: Sequence[tuple[Tensor, ...]],
    ) -> nn.Module: ...


class XnnpackEmbeddingDynamicInt8Strategy:
    """Dynamic per-channel int8 linears plus per-channel int8 embeddings.

    The historical default, validated on the GPT-BERT / NorBERT4 backbone.
    Linear layers become dynamically quantized (per-channel int8 weights,
    runtime-quantized activations); embedding tables become per-channel int8
    (fused into ``embedding_byte`` during lowering). The returned module
    executes in eager PyTorch as the numerical twin of the lowered program, so
    quality gates and fixtures run against it directly.
    """

    def supports_int8(self) -> bool:
        return True

    def prepare_float_adapter(self, adapter: nn.Module) -> nn.Module:
        # NorBERT4 computes each linear weight as ``weight * (scale + 1)`` on
        # every forward; folding that once turns the layer into a plain
        # nn.Linear the quantizer and partitioner understand. A no-op on
        # backbones without scale-parametrized linears (e.g. ModernBERT).
        from prism.exporting.lowering import fold_scaled_linear_parametrizations

        folded = fold_scaled_linear_parametrizations(adapter)
        print(f"Folded {folded} scale-parametrized linears (exact).")
        return adapter

    def quantize(
        self,
        *,
        adapter: nn.Module,
        calibration_batches: Sequence[tuple[Tensor, ...]],
    ) -> nn.Module:
        xnnpack_quantizer_module = _require_executorch_module(
            "executorch.backends.xnnpack.quantizer.xnnpack_quantizer"
        )
        quantize_pt2e = _require_executorch_module(
            "torchao.quantization.pt2e.quantize_pt2e"
        )
        composable = _require_executorch_module(
            "torchao.quantization.pt2e.quantizer.composable_quantizer"
        )
        embedding = _require_executorch_module(
            "torchao.quantization.pt2e.quantizer.embedding_quantizer"
        )

        adapter.eval()
        training_module = torch.export.export(
            adapter,
            tuple(calibration_batches[0]),
            strict=True,
        ).module()
        xnnpack_quantizer = xnnpack_quantizer_module.XNNPACKQuantizer()
        xnnpack_quantizer.set_global(
            xnnpack_quantizer_module.get_symmetric_quantization_config(
                is_per_channel=True,
                is_dynamic=True,
            )
        )
        quantizer = composable.ComposableQuantizer(
            [embedding.EmbeddingQuantizer(), xnnpack_quantizer]
        )
        prepared = quantize_pt2e.prepare_pt2e(training_module, quantizer)
        with torch.no_grad():
            for batch in calibration_batches:
                prepared(*batch)
        return quantize_pt2e.convert_pt2e(prepared)


class ModernBertInt8Strategy:
    """int8 for the ModernBERT / Ettin backbone — not yet supported.

    ModernBERT builds its sliding-window attention mask by advanced-indexing
    ``attention_mask`` with ``arange``-derived integer index tensors
    (``attention_mask[arange_i, arange_j + offset]``). Under ``prepare_pt2e``
    the calibration forward floats one of those index tensors and ``aten.index``
    rejects it (``IndexError: tensors used as indices must be long, int, byte or
    bool tensors``). The integer index nodes are not themselves annotated by the
    XNNPACK/embedding quantizer, so this is a subtle ``prepare_pt2e``/ModernBERT
    interaction rather than a one-line annotation filter — tracked as a
    follow-up. fp32 export is fully supported and is the shipped path.
    """

    def supports_int8(self) -> bool:
        return False

    def prepare_float_adapter(self, adapter: nn.Module) -> nn.Module:
        return adapter

    def quantize(
        self,
        *,
        adapter: nn.Module,
        calibration_batches: Sequence[tuple[Tensor, ...]],
    ) -> nn.Module:
        raise NotImplementedError(
            "int8 quantization is not yet supported for the ModernBERT/Ettin "
            "backbone: PT2E floats ModernBERT's sliding-window mask index. "
            "Export with --precision fp32; int8 is a tracked follow-up."
        )


_STRATEGIES: dict[str, type] = {
    "xnnpack-embedding-dynamic": XnnpackEmbeddingDynamicInt8Strategy,
    "modernbert": ModernBertInt8Strategy,
}


def resolve_int8_quantization_strategy(name: str) -> Int8QuantizationStrategy:
    """Return the int8 strategy for a profile's ``quantization`` discriminator."""

    try:
        strategy_type = _STRATEGIES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown int8 quantization strategy: {name!r}. "
            f"Known strategies: {', '.join(sorted(_STRATEGIES))}."
        ) from error
    return strategy_type()
