import pytest
import torch
from torch import nn

from prism.data import TokenTaskTargetBatch
from prism.modeling import CharacterTokenBatch, TokenizedBatch, TokenTaskLogits
from prism.schema import MorphologyFeatureSchema, MorphologySchema
from prism.training import (
    RelationDistillationPolicy,
    SupervisedTokenTaskBatch,
    TokenTaskDistillationPolicy,
    TokenTaskLossWeights,
    train_distilled_token_task_step,
)


CATEGORICAL_MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Feature",
            values=("Value",),
            allows_multiple_values=False,
        ),
    ),
)
MULTI_LABEL_MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Feature",
            values=("First", "Second"),
            allows_multiple_values=True,
        ),
    ),
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


class CharacterAwareTinyDistillationModel(TinyDistillationModel):
    def __init__(self) -> None:
        super().__init__()
        self.character_encoder = nn.Identity()
        self.received_character_inputs = False

    def forward(
        self,
        batch: TokenizedBatch,
        character_batch: CharacterTokenBatch,
    ) -> TokenTaskLogits:
        self.received_character_inputs = True
        return super().forward(batch)


class RelationAwareTinyStudent(TinyDistillationModel):
    def __init__(self, token_count: int = 2, hidden_size: int = 4) -> None:
        super().__init__()
        self.pooled = nn.Parameter(torch.randn(token_count, hidden_size))

    def forward_with_pooled_states(
        self,
        batch: TokenizedBatch,
    ) -> tuple[torch.Tensor, TokenTaskLogits]:
        pooled = self.pooled.expand(batch.batch_size, -1, -1)
        return pooled, super().forward(batch)


class TinyRelationTeacher(nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = nn.Parameter(hidden_states)

    def encode_pooled_token_states(
        self,
        batch: TokenizedBatch,
    ) -> torch.Tensor:
        return self.hidden_states


def _two_token_batch() -> SupervisedTokenTaskBatch:
    return SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[True, True]]),
            first_subword_indices=torch.tensor([[0, 1]]),
            subword_end_indices=torch.tensor([[1, 2]]),
            token_mask=torch.tensor([[True, True]]),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor([[1, 0]]),
            morphology_targets=(torch.tensor([[[False, True], [True, False]]]),),
            lemma_rule_ids=torch.tensor([[1, 0]]),
            lemma_rule_mask=torch.tensor([[True, True]]),
            token_mask=torch.tensor([[True, True]]),
        ),
    )


def test_distilled_training_step_reports_relation_loss() -> None:
    torch.manual_seed(3)
    student = RelationAwareTinyStudent()
    teacher = TinyDistillationModel()
    relation_teacher = TinyRelationTeacher(torch.randn(1, 2, 8))
    batch = _two_token_batch()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    pooled_before = student.pooled.detach().clone()
    losses = train_distilled_token_task_step(
        student=student,
        teacher=teacher,
        batch=batch,
        optimizer=optimizer,
        max_gradient_norm=1.0,
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        ),
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
        relation_teacher=relation_teacher,
        relation_policy=RelationDistillationPolicy(
            weight=1.0,
            relation_head_count=2,
        ),
    )

    assert losses.relation_loss is not None
    assert torch.isfinite(losses.relation_loss)
    assert not torch.equal(student.pooled.detach(), pooled_before)
    assert relation_teacher.hidden_states.grad is None
    assert not relation_teacher.hidden_states.requires_grad


def test_distilled_training_step_without_relation_teacher_reports_none() -> None:
    student = TinyDistillationModel()
    teacher = TinyDistillationModel()
    batch = _two_token_batch()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.0)

    losses = train_distilled_token_task_step(
        student=student,
        teacher=teacher,
        batch=batch,
        optimizer=optimizer,
        max_gradient_norm=1.0,
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        ),
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
    )

    assert losses.relation_loss is None


def test_distilled_training_step_validates_relation_configuration() -> None:
    student = RelationAwareTinyStudent()
    teacher = TinyDistillationModel()
    batch = _two_token_batch()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.0)

    with pytest.raises(ValueError, match="teacher and its policy"):
        train_distilled_token_task_step(
            student=student,
            teacher=teacher,
            batch=batch,
            optimizer=optimizer,
            max_gradient_norm=1.0,
            distillation_policy=TokenTaskDistillationPolicy.uniform(
                temperature=1.0,
                weight=0.1,
            ),
            morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
            relation_teacher=TinyRelationTeacher(torch.randn(1, 2, 8)),
        )

    plain_student = TinyDistillationModel()
    with pytest.raises(TypeError, match="forward_with_pooled_states"):
        train_distilled_token_task_step(
            student=plain_student,
            teacher=teacher,
            batch=batch,
            optimizer=torch.optim.SGD(plain_student.parameters(), lr=0.0),
            max_gradient_norm=1.0,
            distillation_policy=TokenTaskDistillationPolicy.uniform(
                temperature=1.0,
                weight=0.1,
            ),
            morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
            relation_teacher=TinyRelationTeacher(torch.randn(1, 2, 8)),
            relation_policy=RelationDistillationPolicy(
                weight=1.0,
                relation_head_count=2,
            ),
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
            subword_end_indices=torch.tensor([[1]]),
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
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=2.0,
            weight=0.5,
            categorical_objective="dkd",
            target_class_weight=1.0,
            non_target_class_weight=1.0,
        ),
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
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


def test_distilled_training_step_forwards_character_inputs_to_teacher() -> None:
    student = TinyDistillationModel()
    teacher = CharacterAwareTinyDistillationModel()
    batch = SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.tensor([[True]]),
            first_subword_indices=torch.tensor([[0]]),
            subword_end_indices=torch.tensor([[1]]),
            token_mask=torch.tensor([[True]]),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor([[1]]),
            morphology_targets=(torch.tensor([[[False, True]]]),),
            lemma_rule_ids=torch.tensor([[1]]),
            lemma_rule_mask=torch.tensor([[True]]),
            token_mask=torch.tensor([[True]]),
        ),
        character_inputs=CharacterTokenBatch(
            character_ids=torch.tensor([[[2, 5, 3]]]),
            character_mask=torch.tensor([[[True, True, True]]]),
            token_mask=torch.tensor([[True]]),
        ),
    )
    optimizer = torch.optim.SGD(student.parameters(), lr=0.0)

    train_distilled_token_task_step(
        student=student,
        teacher=teacher,
        batch=batch,
        optimizer=optimizer,
        max_gradient_norm=1.0,
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=2.0,
            weight=0.5,
        ),
        morphology_schema=CATEGORICAL_MORPHOLOGY_SCHEMA,
    )

    assert teacher.received_character_inputs


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
            subword_end_indices=torch.tensor([[1]]),
            token_mask=torch.tensor([[True]]),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor([[0]]),
            morphology_targets=(
                torch.tensor(
                    [[[False, True, False]]],
                    dtype=torch.bool,
                ),
            ),
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
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        ),
        morphology_schema=MULTI_LABEL_MORPHOLOGY_SCHEMA,
    )
    weighted_losses = train_distilled_token_task_step(
        student=weighted_student,
        teacher=teacher,
        batch=batch,
        optimizer=weighted_optimizer,
        max_gradient_norm=1.0,
        distillation_policy=TokenTaskDistillationPolicy.uniform(
            temperature=1.0,
            weight=0.1,
        ),
        morphology_schema=MULTI_LABEL_MORPHOLOGY_SCHEMA,
        loss_weights=TokenTaskLossWeights(
            morphology_weights=(torch.tensor([3.0, 1.0]),),
        ),
    )

    torch.testing.assert_close(
        weighted_losses.distillation_losses.morphology_loss,
        unweighted_losses.distillation_losses.morphology_loss * 2.0,
    )
