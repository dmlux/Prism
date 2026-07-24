import torch
import pytest

from prism.data import TokenTaskTargetBatch
from prism.modeling import TokenTaskLogits
from prism.schema import (
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    build_lemma_rule_schema,
    build_upos_schema,
)
from prism.training import (
    GradientConflictAuditAccumulator,
    TokenTaskRankAuditAccumulator,
    evenly_spaced_batch_indices,
)


def _schema() -> TokenTaskSchema:
    return TokenTaskSchema(
        upos=build_upos_schema(("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Gender",
                    values=("Masc",),
                    allows_multiple_values=False,
                ),
            ),
        ),
        lemma_rules=build_lemma_rule_schema(
            (
                ("cats", "cat"),
                ("ran", "run"),
            )
        ),
    )


def test_rank_audit_attributes_bundle_errors_and_groups_lemma_ranks() -> None:
    schema = _schema()
    candidate_targets = (
        torch.tensor(
            (
                (True, False),
                (False, True),
            ),
            dtype=torch.bool,
        ),
    )
    accumulator = TokenTaskRankAuditAccumulator(
        schema=schema,
        candidate_morphology_targets=candidate_targets,
        lemma_rule_training_counts=(1, 7),
    )
    targets = TokenTaskTargetBatch(
        upos_ids=torch.tensor([[0, 1]], dtype=torch.long),
        morphology_targets=(
            torch.tensor(
                (
                    (
                        (True, False),
                        (False, True),
                    ),
                ),
                dtype=torch.bool,
            ),
        ),
        lemma_rule_ids=torch.tensor([[0, 1]], dtype=torch.long),
        lemma_rule_mask=torch.tensor([[True, True]], dtype=torch.bool),
        token_mask=torch.tensor([[True, True]], dtype=torch.bool),
    )
    accumulator.add(
        logits=TokenTaskLogits(
            upos_logits=torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
            morphology_logits=(
                torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]),
            ),
            lemma_rule_logits=torch.tensor([[[0.0, 2.0], [0.0, 2.0]]]),
            morphology_bundle_scores=torch.tensor(
                [[[3.0, 1.0], [3.0, 1.0]]]
            ),
        ),
        targets=targets,
        rare_mask=torch.tensor([[True, False]], dtype=torch.bool),
        oov_mask=torch.tensor([[False, True]], dtype=torch.bool),
    )

    bundles, lemma = accumulator.finish()

    assert bundles.token_count == 2
    assert bundles.candidate_covered_count == 2
    assert bundles.final_bundle_correct_count == 1
    assert bundles.final_bundle_error_count == 1
    assert bundles.uncovered_error_count == 0
    assert bundles.ranking_error_count == 1
    assert bundles.refinement_error_count == 0
    assert bundles.covered_ranks.top1_accuracy == 0.5
    assert bundles.error_ranks is not None
    assert bundles.error_ranks.mean_rank == 2.0

    assert lemma.overall.top1_accuracy == 0.5
    assert lemma.overall.top2_accuracy == 1.0
    assert tuple(group.name for group in lemma.by_rule_frequency) == (
        "singleton",
        "6-20",
    )
    assert tuple(group.name for group in lemma.by_token_frequency) == (
        "rare",
        "oov",
    )
    assert tuple(group.name for group in lemma.by_upos) == ("NOUN", "VERB")


def test_gradient_audit_reports_opposed_and_aligned_tasks() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    accumulator = GradientConflictAuditAccumulator(
        parameter_groups={"shared": (shared,)},
    )

    accumulator.add(
        upos_loss=shared.sum(),
        morphology_loss=-shared.sum(),
        lemma_loss=(shared * torch.tensor([2.0, 2.0])).sum(),
    )

    group = accumulator.finish()[0]
    pairs = {
        (pair.first_task, pair.second_task): pair for pair in group.task_pairs
    }
    assert pairs[("upos", "morphology")].mean_cosine_similarity == pytest.approx(
        -1.0
    )
    assert pairs[("upos", "morphology")].conflict_rate == 1.0
    assert pairs[("upos", "lemma")].mean_cosine_similarity == pytest.approx(1.0)
    assert pairs[("upos", "lemma")].conflict_rate == 0.0
    assert pairs[("morphology", "lemma")].mean_cosine_similarity == pytest.approx(
        -1.0
    )


def test_evenly_spaced_batch_indices_cover_the_run() -> None:
    assert evenly_spaced_batch_indices(
        batch_count=10,
        selected_count=4,
    ) == (0, 3, 6, 9)
    assert evenly_spaced_batch_indices(
        batch_count=10,
        selected_count=1,
    ) == (5,)
