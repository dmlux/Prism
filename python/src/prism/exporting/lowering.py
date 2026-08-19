"""Lowering of export adapters to portable ExecuTorch programs.

The export dependency stays optional: training and evaluation environments do
not need ExecuTorch, so the imports happen lazily inside the functions and
fail with the install command instead of an import traceback.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from prism.exporting.quantization import Int8QuantizationStrategy


def _require_executorch_module(module_path: str) -> object:
    import importlib

    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Model export requires the optional ExecuTorch dependency; "
            "install it with: python -m pip install -e './python[export]'"
        ) from error


def fold_scaled_linear_parametrizations(root: nn.Module) -> int:
    """Replace scale-parametrized linears with plain, pre-folded nn.Linear.

    The NorBERT4 backbone computes every linear weight as
    ``weight * (scale + 1)`` on each forward (``CastedLinearIn``) and
    derives the fused query/key weights by concatenating parameter lists
    (``MultiCastedLinearOrthoIn``). Folding performs the identical
    multiplication exactly once, so the replacement is numerically exact
    (the forward computes the same product from the same operands) while
    quantizers and delegate partitioners see standard static weights.
    """

    replaced = 0
    for _, parent in list(root.named_modules()):
        for child_name, child in list(parent.named_children()):
            type_name = type(child).__name__
            if type_name == "CastedLinearIn":
                with torch.no_grad():
                    folded = child.weight * (child.scale + 1.0).unsqueeze(0)
            elif type_name == "MultiCastedLinearOrthoIn":
                with torch.no_grad():
                    folded = torch.cat(list(child.weights), dim=0) * (
                        child.scale + 1.0
                    ).unsqueeze(0)
            else:
                continue
            linear = nn.Linear(
                child.in_features,
                folded.shape[0],
                bias=child.bias is not None,
            )
            with torch.no_grad():
                linear.weight.copy_(folded)
                if child.bias is not None:
                    linear.bias.copy_(child.bias)
            setattr(parent, child_name, linear)
            replaced += 1
    return replaced


_PORTABLE_QUANT_OUT_VARIANTS_REGISTERED = False
_PORTABLE_QUANT_OUT_VARIANTS_LIB = None


def _ensure_portable_quant_out_variants() -> None:
    """Register Python ``.out`` overloads for the per-channel quant ops.

    ExecuTorch ships portable out-variant kernels for
    ``quantized_decomposed::{quantize,dequantize}_per_channel`` (see
    ``executorch/kernels/quantized/quantized.yaml``) for its C++ runtime, but
    does NOT register them as Python torch ops, so the AOT ``to_out_var`` pass
    cannot resolve them: ``to_executorch`` then fails with "Missing out variants:
    quantized_decomposed::quantize_per_channel" whenever a per-channel quantize
    is left un-delegated. The RoPE backbone never triggers this (its embedding
    fuses into ``embedding_byte`` and every linear delegates to XNNPACK), but the
    BabyLM GPT-BERT backbone does: its DeBERTa-style disentangled relative
    attention applies a linear to the constant ``relative_embedding``, which the
    XNNPACK dynamic-quant partitioner does not claim, leaving a per-channel
    weight quantize in the graph.

    Registering the ``.out`` overloads here (matching the shipped yaml schema, and
    forwarding to the functional op) lets the pass complete; the op executes on
    the portable backend at runtime using the same kernel the yaml declares. It
    is a no-op for graphs that leave no un-delegated per-channel quantize.
    Idempotent; the Library is held alive at module scope.
    """
    global _PORTABLE_QUANT_OUT_VARIANTS_REGISTERED, _PORTABLE_QUANT_OUT_VARIANTS_LIB
    if _PORTABLE_QUANT_OUT_VARIANTS_REGISTERED:
        return
    import torch.ao.quantization.fx._decomposed  # noqa: F401 (registers functional ops)

    quantized_decomposed = torch.ops.quantized_decomposed
    library = torch.library.Library("quantized_decomposed", "FRAGMENT")

    if "out" not in quantized_decomposed.quantize_per_channel.overloads():
        library.define(
            "quantize_per_channel.out(Tensor input, Tensor scales, "
            "Tensor zero_points, int axis, int quant_min, int quant_max, "
            "ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
        )

        def _quantize_per_channel_out(
            input, scales, zero_points, axis, quant_min, quant_max, dtype, *, out
        ):
            out.copy_(
                quantized_decomposed.quantize_per_channel.default(
                    input, scales, zero_points, axis, quant_min, quant_max, dtype
                )
            )
            return out

        library.impl(
            "quantize_per_channel.out",
            _quantize_per_channel_out,
            "CompositeExplicitAutograd",
        )

    if "out" not in quantized_decomposed.dequantize_per_channel.overloads():
        library.define(
            "dequantize_per_channel.out(Tensor input, Tensor scales, "
            "Tensor? zero_points, int axis, int quant_min, int quant_max, "
            "ScalarType dtype, *, ScalarType? out_dtype=None, Tensor(a!) out) "
            "-> Tensor(a!)"
        )

        def _dequantize_per_channel_out(
            input, scales, zero_points, axis, quant_min, quant_max, dtype,
            *, out_dtype=None, out,
        ):
            out.copy_(
                quantized_decomposed.dequantize_per_channel.default(
                    input, scales, zero_points, axis, quant_min, quant_max, dtype
                )
            )
            return out

        library.impl(
            "dequantize_per_channel.out",
            _dequantize_per_channel_out,
            "CompositeExplicitAutograd",
        )

    _PORTABLE_QUANT_OUT_VARIANTS_LIB = library
    _PORTABLE_QUANT_OUT_VARIANTS_REGISTERED = True


def quantize_adapter_int8(
    *,
    adapter: nn.Module,
    calibration_batches: Sequence[tuple[Tensor, ...]],
    strategy: "Int8QuantizationStrategy | None" = None,
) -> nn.Module:
    """Quantize an export adapter to int8 and return its eager twin.

    Delegates to the backbone's :class:`Int8QuantizationStrategy`; the default
    reproduces the historical XNNPACK dynamic-linear + per-channel-embedding
    path (GPT-BERT / NorBERT4). The returned module executes in eager PyTorch as
    the numerical twin of the lowered program, so quality gates and fixtures run
    against it directly.
    """

    if strategy is None:
        from prism.exporting.quantization import XnnpackEmbeddingDynamicInt8Strategy

        strategy = XnnpackEmbeddingDynamicInt8Strategy()
    return strategy.quantize(
        adapter=adapter,
        calibration_batches=calibration_batches,
    )


def lower_to_executorch_xnnpack(
    *,
    adapter: nn.Module,
    example_inputs: tuple[Tensor, ...],
    external_data_name: str | None = None,
    quantized: bool = False,
    int8_strategy: "Int8QuantizationStrategy | None" = None,
) -> "LoweredXnnpackProgram":
    """Capture an export adapter strictly and lower it to an XNNPACK program.

    With ``external_data_name``, the delegate's constant weights are tagged
    for program-data separation: the returned program references them by
    content hash and ``write_data_files`` writes them into a shared
    ``<name>.ptd`` file. Programs lowered from the same adapter share the
    hashes, so several fixed-shape programs can load one data file.

    With ``quantized``, the adapter must be the floating (folded) module;
    it is quantized here at the example shapes, because converted PT2E
    twins carry shape guards and cannot be re-exported for other fixed
    shapes. Weight quantization is deterministic (dynamic linears,
    weight-only embeddings), so every shape produces byte-identical
    weights and the shared data file stays valid across programs.
    Dynamically quantized linears and the surrounding fp32 ops are delegated
    by a single grouped XNNPACK partitioner (both dynamic-quant and fp32
    precisions), which produces few large delegate subgraphs instead of one
    per op. The earlier per-op workaround (the grouped two-partitioner combo
    tripped over pass-through arguments) is no longer needed and was ~5-16%
    slower from micro-delegate overhead; the embedding lookup is fused into
    ``embedding_byte``, which requires the quantized kernel library at runtime.
    """

    xnnpack_partitioner = _require_executorch_module(
        "executorch.backends.xnnpack.partition.xnnpack_partitioner"
    )
    exir = _require_executorch_module("executorch.exir")

    if quantized:
        adapter = quantize_adapter_int8(
            adapter=adapter,
            calibration_batches=(example_inputs,),
            strategy=int8_strategy,
        )
    else:
        adapter.eval()
    exported_program = torch.export.export(
        adapter,
        example_inputs,
        strict=True,
    )
    backend_passes = []
    external_constants = None
    if external_data_name is not None:
        # Two complementary passes cover every weight: the delegate pass
        # tags constants consumed by the XNNPACK payload (linear weights),
        # and the backend-config callable tags the constants the delegate
        # leaves in the program (most importantly the subword embedding).
        external_constants_pass = _require_executorch_module(
            "executorch.exir.passes.external_constants_pass"
        )
        unlifted = exported_program.module()
        external_constants_pass.delegate_external_constants_pass_unlifted(
            module=unlifted,
            gen_tag_fn=lambda _node: external_data_name,
        )
        exported_program = torch.export.export(
            unlifted,
            example_inputs,
            strict=True,
        )
        external_constants = lambda _node: external_data_name  # noqa: E731

    if quantized:
        # Make the portable per-channel quant out-variants resolvable by the AOT
        # to_out_var pass, so backbones that leave an un-delegated per-channel
        # quantize (BabyLM GPT-BERT's relative-position path) still lower; no-op
        # for fully-delegated graphs (RoPE).
        _ensure_portable_quant_out_variants()
        xnnpack_config = _require_executorch_module(
            "executorch.backends.xnnpack.partition.config.xnnpack_config"
        )
        quant_fusion = _require_executorch_module(
            "executorch.exir.passes.quant_fusion_pass"
        )
        # A single grouped partitioner that owns both the dynamic-quant
        # linears and the surrounding fp32 ops. Grouping (per_op_mode=False)
        # yields few large delegate subgraphs; per-op fragmentation added
        # ~5-16% XNNPACK call overhead (measured NorBERT4 + ModernBERT). Both
        # precisions must live in ONE partitioner: a two-partitioner grouped
        # combo trips the "output node already in inputs" lowering error.
        partitioners = [
            xnnpack_partitioner.XnnpackPartitioner(
                config_precisions=[
                    xnnpack_config.ConfigPrecisionType.DYNAMIC_QUANT,
                    xnnpack_config.ConfigPrecisionType.FP32,
                ],
                per_op_mode=False,
            ),
        ]
        backend_passes.append(quant_fusion.QuantFusionPass())
    else:
        partitioners = [xnnpack_partitioner.XnnpackPartitioner()]

    backend_config = exir.ExecutorchBackendConfig(
        passes=backend_passes,
        **(
            {"external_constants": external_constants}
            if external_constants is not None
            else {}
        ),
    )
    executorch_program = exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=partitioners,
    ).to_executorch(backend_config)

    return LoweredXnnpackProgram(
        program_bytes=bytes(executorch_program.buffer),
        _executorch_program=executorch_program,
    )


class LoweredXnnpackProgram:
    """A serialized program plus its optional external tensor data."""

    def __init__(self, *, program_bytes: bytes, _executorch_program: object) -> None:
        self.program_bytes = program_bytes
        self._executorch_program = _executorch_program

    def write_data_files(self, output_directory) -> None:
        """Write the tagged external data as ``<name>.ptd`` files."""

        self._executorch_program.write_tensor_data_to_file(str(output_directory))


def run_executorch_program(
    *,
    program_bytes: bytes,
    inputs: Sequence[Tensor],
    data_path: object | None = None,
) -> tuple[Tensor, ...]:
    """Execute a serialized ExecuTorch program's forward method.

    ``data_path`` names the ``.ptd`` file holding externally stored weights;
    loading it requires the file-based path because the high-level runtime
    API does not accept external data for in-memory programs.
    """

    if data_path is not None:
        import tempfile
        from pathlib import Path

        portable_lib = _require_executorch_module(
            "executorch.extension.pybindings.portable_lib"
        )
        with tempfile.TemporaryDirectory() as staging_name:
            program_path = Path(staging_name) / "program.pte"
            program_path.write_bytes(program_bytes)
            module = portable_lib._load_for_executorch(
                str(program_path),
                data_path=str(data_path),
            )
            outputs = module.forward(list(inputs))
    else:
        runtime_module = _require_executorch_module("executorch.runtime")

        runtime = runtime_module.Runtime.get()
        program = runtime.load_program(program_bytes)
        method = program.load_method("forward")
        if method is None:
            raise ValueError("ExecuTorch program does not provide a forward method.")
        outputs = method.execute(list(inputs))

    if not isinstance(outputs, (list, tuple)) or not all(
        isinstance(output, Tensor) for output in outputs
    ):
        raise ValueError("ExecuTorch execution must return tensors.")

    return tuple(outputs)


def maximum_absolute_difference(
    reference_outputs: Sequence[Tensor],
    candidate_outputs: Sequence[Tensor],
) -> float:
    """Compare two output tuples and return the largest elementwise gap."""

    if len(reference_outputs) != len(candidate_outputs):
        raise ValueError(
            "Parity comparison requires the same number of output tensors: "
            f"{len(reference_outputs)} versus {len(candidate_outputs)}."
        )

    largest = 0.0
    for reference, candidate in zip(
        reference_outputs,
        candidate_outputs,
        strict=True,
    ):
        if reference.shape != candidate.shape:
            raise ValueError(
                "Parity comparison requires matching output shapes: "
                f"{tuple(reference.shape)} versus {tuple(candidate.shape)}."
            )
        difference = (reference - candidate).abs().max().item()
        largest = max(largest, float(difference))

    return largest
