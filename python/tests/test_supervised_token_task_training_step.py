import math

import torch
from torch import nn

from prism.data import TokenTaskTargetBatch
from prism.modeling import (
    TokenizedBatch,
    TokenTaskLogits,
)
from prism.training import (
    SupervisedTokenTaskBatch,
    TokenTaskLossWeights,
    evaluate_supervised_token_task_step,
    train_supervised_token_task_step,
)


class TinyTokenTaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.upos = nn.Parameter(torch.zeros(2))
        self.morphology = nn.Parameter(torch.zeros(2))
        self.lemma_rules = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        batch: TokenizedBatch,
    ) -> TokenTaskLogits:
        token_dimensions = (
            batch.batch_size,
            batch.max_token_count,
            2,
        )

        return TokenTaskLogits(
            upos_logits=self.upos.expand(token_dimensions),
            morphology_logits=(self.morphology.expand(token_dimensions),),
            lemma_rule_logits=self.lemma_rules.expand(token_dimensions),
        )


def test_training_step_updates_model_parameters() -> None:
    model = TinyTokenTaskModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    batch = SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor(
                [[1, 2, 3]],
                dtype=torch.long,
            ),
            attention_mask=torch.tensor(
                [[True, True, True]],
                dtype=torch.bool,
            ),
            first_subword_indices=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            morphology_targets=(
                torch.tensor(
                    [[[False, True]]],
                    dtype=torch.bool,
                ),
            ),
            lemma_rule_ids=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            lemma_rule_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
    )
    previous_upos = model.upos.detach().clone()

    loss_weights = TokenTaskLossWeights(
        morphology_positive_weights=(torch.tensor([1.0, 3.0]),),
    )

    losses = train_supervised_token_task_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        max_gradient_norm=0.01,
        loss_weights=loss_weights,
    )

    torch.testing.assert_close(
        losses.morphology_loss,
        torch.tensor(2.0 * math.log(2.0)),
    )

    assert torch.isfinite(losses.total_loss)

    gradients = tuple(
        gradient
        for parameter in model.parameters()
        if (gradient := parameter.grad) is not None
    )
    total_gradient_norm = torch.linalg.vector_norm(
        torch.cat(tuple(gradient.detach().flatten() for gradient in gradients))
    )

    assert total_gradient_norm.item() <= 0.010001

    assert not torch.equal(
        model.upos.detach(),
        previous_upos,
    )

    parameters_before_evaluation = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    optimizer.zero_grad(set_to_none=True)

    evaluation_logits, evaluation_losses = evaluate_supervised_token_task_step(
        model=model,
        batch=batch,
    )

    assert isinstance(evaluation_logits, TokenTaskLogits)
    assert not model.training
    assert not evaluation_losses.total_loss.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())

    for parameter, previous_parameter in zip(
        model.parameters(),
        parameters_before_evaluation,
        strict=True,
    ):
        torch.testing.assert_close(parameter, previous_parameter)
