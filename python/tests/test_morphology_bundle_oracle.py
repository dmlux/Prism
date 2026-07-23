import pytest

from prism.evaluation import (
    MorphologyBundleExample,
    MorphologyBundleInventory,
    evaluate_morphology_bundle_oracle,
)


def _example(upos: str, *features: str) -> MorphologyBundleExample:
    return MorphologyBundleExample(upos=upos, bundle=tuple(sorted(features)))


def test_morphology_bundle_oracle_reports_frequency_ranked_coverage() -> None:
    inventory = MorphologyBundleInventory.from_examples(
        (
            _example("NOUN", "Number=Sing"),
            _example("NOUN", "Number=Sing"),
            _example("NOUN", "Number=Plur"),
            _example("ADJ", "Degree=Pos"),
            _example("PUNCT"),
        )
    )

    metrics = evaluate_morphology_bundle_oracle(
        inventory=inventory,
        development_examples=(
            _example("NOUN", "Number=Sing"),
            _example("NOUN", "Number=Plur"),
            _example("ADJ", "Degree=Cmp"),
            _example("PUNCT"),
        ),
        candidate_counts=(1, 2),
    )

    assert metrics.training_example_count == 5
    assert metrics.distinct_bundle_count == 4
    assert metrics.distinct_upos_bundle_count == 4
    assert metrics.maximum_candidate_count == 2
    assert metrics.overall.token_count == 4
    assert metrics.overall.globally_seen_token_count == 3
    assert metrics.overall.gold_upos_seen_token_count == 3
    assert metrics.overall.top_k[0].covered_token_count == 2
    assert metrics.overall.top_k[1].covered_token_count == 3
    assert metrics.annotated.token_count == 3


def test_morphology_bundle_oracle_rejects_invalid_candidate_counts() -> None:
    inventory = MorphologyBundleInventory.from_examples((_example("NOUN"),))

    with pytest.raises(ValueError, match="unique, sorted, and positive"):
        evaluate_morphology_bundle_oracle(
            inventory=inventory,
            development_examples=(_example("NOUN"),),
            candidate_counts=(2, 1),
        )
