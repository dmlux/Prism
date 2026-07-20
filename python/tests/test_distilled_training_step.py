import torch
from torch import nn

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenizedBatch, TokenTaskLogits
from prism.training import (
    SupervisedTokenTaskBatch,
    TokenTaskLossWeights,
    train_distilled_token_task_step,
)


class TinyDistillationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upos = nn.Parameter(torch.zeros(2))
        self.morphology = nn.Parameter(torch.zeros(2))
        self.lemma_rules = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        batch: TokenizedBatch,
    ) -> TokenTaskLogits:
        dimensions = (
            batch.batch_size,
            batch.max_token_count,
            2,
        )

        return TokenTaskLogits(
            upos_logits=self.upos.expand(dimensions),
            morphology_logits=(self.morphology.expand(dimensions),),
            lemma_rule_logits=self.lemma_rules.expand(dimensions),
        )


def test_distilled_training_step_only_updates_student() -> None:
    student = TinyDistillationModel()
    teacher = TinyDistillationModel()

    with torch.no_grad():
        teacher.upos.copy_(torch.tensor([2.0, -2.0]))
        teacher.morphology.copy_(torch.tensor([1.0, -1.0]))
        teacher.lemma_rules.copy_(torch.tensor([-2.0, 2.0]))

    batch = SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.tensor([[True]]),
            first_subword_indices=torch.tensor([[0]]),
            token_mask=torch.tensor([[True]]),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor([[1]]),
            morphology_targets=(torch.tensor([[[False, True]]]),),
            lemma_rule_ids=torch.tensor([[1]]),
            lemma_rule_mask=torch.tensor([[True]]),
            token_mask=torch.tensor([[True]]),
        ),
    )
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=0.1,
    )

    student_parameters_before = tuple(
        parameter.detach().clone() for parameter in student.parameters()
    )
    teacher_parameters_before = tuple(
        parameter.detach().clone() for parameter in teacher.parameters()
    )

    losses = train_distilled_token_task_step(
        student=student,
        teacher=teacher,
        batch=batch,
        optimizer=optimizer,
        max_gradient_norm=1.0,
        temperature=2.0,
        distillation_weight=0.5,
    )

    assert torch.isfinite(losses.total_loss)
    assert not teacher.training
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(
        not torch.equal(parameter, previous)
        for parameter, previous in zip(
            student.parameters(),
            student_parameters_before,
            strict=True,
        )
    )

    for parameter, previous in zip(
        teacher.parameters(),
        teacher_parameters_before,
        strict=True,
    ):
        torch.testing.assert_close(parameter, previous)


def test_distilled_training_step_forwards_morphology_weights() -> None:
    unweighted_student = TinyDistillationModel()
    weighted_student = TinyDistillationModel()
    teacher = TinyDistillationModel()

    with torch.no_grad():
        teacher.morphology.copy_(torch.tensor([1.0, -1.0]))

    batch = SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.tensor([[True]]),
            first_subword_indices=torch.tensor([[0]]),
            token_mask=torch.tensor([[True]]),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor([[0]]),
            morphology_targets=(torch.tensor([[[False, True]]]),),
            lemma_rule_ids=torch.tensor([[0]]),
            lemma_rule_mask=torch.tensor([[True]]),
            token_mask=torch.tensor([[True]]),
        ),
    )
    unweighted_optimizer = torch.optim.SGD(
        unweighted_student.parameters(),
        lr=0.0,
    )
    weighted_optimizer = torch.optim.SGD(
        weighted_student.parameters(),
        lr=0.0,
    )

    unweighted_losses = train_distilled_token_task_step(
        student=unweighted_student,
        teacher=teacher,
        batch=batch,
        optimizer=unweighted_optimizer,
        max_gradient_norm=1.0,
        temperature=1.0,
        distillation_weight=0.1,
    )
    weighted_losses = train_distilled_token_task_step(
        student=weighted_student,
        teacher=teacher,
        batch=batch,
        optimizer=weighted_optimizer,
        max_gradient_norm=1.0,
        temperature=1.0,
        distillation_weight=0.1,
        loss_weights=TokenTaskLossWeights(
            morphology_positive_weights=(torch.tensor([1.0, 3.0]),),
        ),
    )

    torch.testing.assert_close(
        weighted_losses.distillation_losses.morphology_loss,
        unweighted_losses.distillation_losses.morphology_loss * 2.0,
    )
