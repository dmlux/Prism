"""Backbone-specific int8 quantization strategies for artifact export.

Quantization is architecture-specific, so it lives behind a language- and
backbone-independent interface instead of being wired into the shared export
flow. Each backbone family supplies its own strategy:

* the GPT-BERT / NorBERT4 graph quantizes cleanly with the standard XNNPACK
  dynamic-linear + per-channel-embedding PT2E path
  (:class:`XnnpackEmbeddingDynamicInt8Strategy`, the historical default);
* ModernBERT / Ettin uses the SAME PT2E path but first swaps its transformers
  attention-mask builders for export-clean equivalents during capture
  (:class:`ModernBertInt8Strategy`): the stock builders index ``attention_mask``
  with ``arange``-derived integers, which PT2E's convert bakes a dequantize
  onto and floats, so the converted graph is not runnable/lowerable. The
  replacements produce a bit-identical mask with only broadcast + comparison.

The fp32 export path never touches this module, so it is unchanged for every
backbone. A profile selects its strategy by a string discriminator
(``LanguageProfileSpec.quantization``); :func:`resolve_int8_quantization_strategy`
maps that to an instance at export time, keeping ``prism.exporting`` free of any
dependency on ``prism.languages``.
"""

import contextlib
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


def _clean_bidirectional_kv_mask(
    attention_mask: "Tensor | None",
    batch_size: int,
    query_length: int,
    device: object,
) -> Tensor:
    """Keep-mask ``(batch, 1, 1, kv)`` from the 2D padding mask, no indexing."""

    if attention_mask is None:
        return torch.ones(
            batch_size, 1, 1, query_length, dtype=torch.bool, device=device
        )
    return attention_mask.to(torch.bool)[:, None, None, :]


def _additive(keep: Tensor, dtype: torch.dtype, device: object) -> Tensor:
    return torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
    )


def _clean_bidirectional_mask(config, inputs_embeds, attention_mask, **kwargs):
    """Export-clean twin of transformers' ``create_bidirectional_mask``.

    Verified bit-identical to the stock builder (broadcast padding only, since
    a bidirectional full mask attends everywhere).
    """

    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    batch_size, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
    keep = _clean_bidirectional_kv_mask(
        attention_mask, batch_size, query_length, inputs_embeds.device
    ).expand(batch_size, 1, query_length, query_length)
    return _additive(keep, inputs_embeds.dtype, inputs_embeds.device)


def _clean_sliding_window_mask(config, inputs_embeds, attention_mask, **kwargs):
    """Export-clean twin of ``create_bidirectional_sliding_window_mask``.

    The stock sliding overlay is already arithmetic (``|q-kv| <= window``); the
    padding is applied by broadcast instead of ``attention_mask[b, kv]`` fancy
    indexing. Verified bit-identical to the stock builder.
    """

    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    batch_size, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
    device, dtype = inputs_embeds.device, inputs_embeds.dtype
    sliding_window = getattr(config, "sliding_window")
    positions = torch.arange(query_length, device=device)
    band = (positions[:, None] - positions[None, :]).abs() <= sliding_window
    keep = band[None, None, :, :] & _clean_bidirectional_kv_mask(
        attention_mask, batch_size, query_length, device
    )
    return _additive(keep, dtype, device)


@contextlib.contextmanager
def _modernbert_export_clean_masks():
    """Swap ModernBERT's mask builders for the export-clean twins, then restore.

    Scoped to the strict-export capture only, and only ever entered by
    :class:`ModernBertInt8Strategy` — the NorBERT4 path never touches this, so
    Norwegian export is unaffected.
    """

    import importlib

    modeling = importlib.import_module(
        "transformers.models.modernbert.modeling_modernbert"
    )
    saved = (
        modeling.create_bidirectional_mask,
        modeling.create_bidirectional_sliding_window_mask,
    )
    modeling.create_bidirectional_mask = _clean_bidirectional_mask
    modeling.create_bidirectional_sliding_window_mask = _clean_sliding_window_mask
    try:
        yield
    finally:
        (
            modeling.create_bidirectional_mask,
            modeling.create_bidirectional_sliding_window_mask,
        ) = saved


class ModernBertInt8Strategy:
    """int8 for the ModernBERT / Ettin backbone via the NorBERT4 PT2E recipe.

    Identical dynamic per-channel int8 linears + per-channel int8 embeddings and
    identical ``quantized_decomposed`` → XNNPACK lowering as
    :class:`XnnpackEmbeddingDynamicInt8Strategy`. The only ModernBERT-specific
    step is capturing the strict export under
    :func:`_modernbert_export_clean_masks`, which replaces the stock attention
    mask builders (that index ``attention_mask`` with ``arange`` integers — PT2E
    convert floats that index and breaks the graph) with bit-identical
    broadcast-only equivalents. A warm forward first installs transformers'
    output-capturing hooks, whose lazy lock ``torch.export`` cannot trace.
    """

    def supports_int8(self) -> bool:
        return True

    def prepare_float_adapter(self, adapter: nn.Module) -> nn.Module:
        # No scale-parametrized linears to fold (a NorBERT4 concern).
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
        # Warm transformers' output-capturing hooks (installed lazily under a
        # threading lock that torch.export's dynamo cannot trace) before capture.
        if calibration_batches:
            with torch.no_grad():
                adapter(*calibration_batches[0])
        with _modernbert_export_clean_masks():
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
