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
    external_data_name: str | None = None,
) -> "LoweredXnnpackProgram":
    """Capture an export adapter strictly and lower it to an XNNPACK program.

    With ``external_data_name``, the delegate's constant weights are tagged
    for program-data separation: the returned program references them by
    content hash and ``write_data_files`` writes them into a shared
    ``<name>.ptd`` file. Programs lowered from the same adapter share the
    hashes, so several fixed-shape programs can load one data file.
    """

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
    backend_config = exir.ExecutorchBackendConfig()
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
        backend_config = exir.ExecutorchBackendConfig(
            external_constants=lambda _node: external_data_name,
        )
    executorch_program = exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=[xnnpack_partitioner.XnnpackPartitioner()],
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
