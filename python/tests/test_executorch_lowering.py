import pytest
import torch
from torch import nn

from prism.exporting import (
    TokenTaggerExportAdapter,
    lower_to_executorch_xnnpack,
    maximum_absolute_difference,
    run_executorch_program,
)
from prism.modeling import TokenizedBatch
from prism.modeling.outputs import TokenTaskLogits

pytest.importorskip("executorch")


class TinyLinearTokenTagger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upos_head = nn.Linear(1, 2)
        self.lemma_head = nn.Linear(1, 3)

    def forward(self, batch: TokenizedBatch) -> TokenTaskLogits:
        hidden_states = batch.input_ids.to(dtype=torch.float32).unsqueeze(-1)

        return TokenTaskLogits(
            upos_logits=self.upos_head(hidden_states),
            morphology_logits=(hidden_states + 2.0,),
            lemma_rule_logits=self.lemma_head(hidden_states),
        )


def _example_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        torch.tensor([[True, True, True, False]]),
        torch.tensor([[1, 2, 0]], dtype=torch.long),
        torch.tensor([[2, 3, 0]], dtype=torch.long),
        torch.tensor([[True, True, False]]),
    )


def test_lowered_program_matches_eager_outputs() -> None:
    adapter = TokenTaggerExportAdapter(model=TinyLinearTokenTagger())
    inputs = _example_inputs()

    with torch.no_grad():
        eager_outputs = adapter(*inputs)
    program_bytes = lower_to_executorch_xnnpack(
        adapter=adapter,
        example_inputs=inputs,
    )
    runtime_outputs = run_executorch_program(
        program_bytes=program_bytes,
        inputs=inputs,
    )

    assert maximum_absolute_difference(eager_outputs, runtime_outputs) < 1e-5


def test_maximum_absolute_difference_requires_matching_outputs() -> None:
    with pytest.raises(ValueError, match="number of output tensors"):
        maximum_absolute_difference((torch.zeros(2),), ())
    with pytest.raises(ValueError, match="matching output shapes"):
        maximum_absolute_difference((torch.zeros(2),), (torch.zeros(3),))
