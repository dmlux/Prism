"""Lowering of export adapters to portable ExecuTorch programs.

The export dependency stays optional: training and evaluation environments do
not need ExecuTorch, so the imports happen lazily inside the functions and
fail with the install command instead of an import traceback.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _require_executorch_module(module_path: str) -> object:
    import importlib

    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Model export requires the optional ExecuTorch dependency; "
            "install it with: python -m pip install -e './python[export]'"
        ) from error


def lower_to_executorch_xnnpack(
    *,
    adapter: nn.Module,
    example_inputs: tuple[Tensor, ...],
) -> bytes:
    """Capture an export adapter strictly and lower it to an XNNPACK program."""

    xnnpack_partitioner = _require_executorch_module(
        "executorch.backends.xnnpack.partition.xnnpack_partitioner"
    )
    exir = _require_executorch_module("executorch.exir")

    adapter.eval()
    exported_program = torch.export.export(
        adapter,
        example_inputs,
        strict=True,
    )
    executorch_program = exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=[xnnpack_partitioner.XnnpackPartitioner()],
    ).to_executorch()

    return bytes(executorch_program.buffer)


def run_executorch_program(
    *,
    program_bytes: bytes,
    inputs: Sequence[Tensor],
) -> tuple[Tensor, ...]:
    """Execute a serialized ExecuTorch program's forward method."""

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
