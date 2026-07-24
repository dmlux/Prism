import torch

from prism.schema import (
    MorphologyFeatureSchema,
    MorphologySchema,
)
from prism.training import (
    MorphologyHeadProbeArchitecture,
    MorphologyHeadProbeConfig,
    MorphologyProbeDataset,
    MorphologyProbeSlice,
    evaluate_morphology_head_probe,
    train_morphology_head_probe,
)


def _morphology_schema() -> MorphologySchema:
    return MorphologySchema(
        version=1,
        features=(
            MorphologyFeatureSchema(
                name="Gender",
                values=("Fem", "Masc"),
                allows_multiple_values=False,
            ),
            MorphologyFeatureSchema(
                name="Number",
                values=("Plur", "Sing"),
                allows_multiple_values=True,
            ),
        ),
    )


def _probe_dataset() -> MorphologyProbeDataset:
    hidden_states = torch.tensor(
        (
            (-2.0, -1.0, 0.0, 0.0),
            (-1.0, -2.0, 0.0, 0.0),
            (1.0, 2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0, 0.0),
        ),
        dtype=torch.float32,
    ).repeat((16, 1))
    gender_targets = torch.tensor(
        (
            (False, True, False),
            (False, True, False),
            (False, False, True),
            (False, False, True),
        ),
        dtype=torch.bool,
    ).repeat((16, 1))
    number_targets = torch.tensor(
        (
            (False, True, False),
            (False, True, False),
            (False, False, True),
            (False, False, True),
        ),
        dtype=torch.bool,
    ).repeat((16, 1))
    oov_mask = torch.zeros(hidden_states.shape[0], dtype=torch.bool)
    oov_mask[::4] = True
    return MorphologyProbeDataset(
        hidden_states=hidden_states,
        morphology_targets=(gender_targets, number_targets),
        slices=(MorphologyProbeSlice(name="oov", mask=oov_mask),),
    )


def test_frozen_morphology_probe_trains_and_reports_slices() -> None:
    schema = _morphology_schema()
    dataset = _probe_dataset()
    config = MorphologyHeadProbeConfig(
        epoch_count=12,
        batch_size=16,
        learning_rate=0.05,
        weight_decay=0.0,
        dropout_probability=0.0,
        random_seed=7,
    )

    probe, losses = train_morphology_head_probe(
        dataset=dataset,
        morphology_schema=schema,
        architecture=MorphologyHeadProbeArchitecture.LINEAR,
        config=config,
        device=torch.device("cpu"),
    )
    metrics = evaluate_morphology_head_probe(
        probe=probe,
        dataset=dataset,
        morphology_schema=schema,
        device=torch.device("cpu"),
        batch_size=16,
    )

    assert len(losses) == config.epoch_count
    assert losses[-1] < losses[0]
    assert tuple(feature.feature_name for feature in metrics) == (
        "Gender",
        "Number",
    )
    assert metrics[0].overall.accuracy == 1.0
    assert metrics[0].annotated.accuracy == 1.0
    assert metrics[0].slices[0].name == "oov"
    assert metrics[0].slices[0].accuracy.accuracy == 1.0


def test_feature_mlp_probe_adds_feature_specific_capacity() -> None:
    schema = _morphology_schema()
    dataset = _probe_dataset()
    config = MorphologyHeadProbeConfig(
        epoch_count=1,
        batch_size=dataset.token_count,
        dropout_probability=0.0,
    )

    linear, _ = train_morphology_head_probe(
        dataset=dataset,
        morphology_schema=schema,
        architecture=MorphologyHeadProbeArchitecture.LINEAR,
        config=config,
        device=torch.device("cpu"),
    )
    feature_mlp, _ = train_morphology_head_probe(
        dataset=dataset,
        morphology_schema=schema,
        architecture=MorphologyHeadProbeArchitecture.FEATURE_MLP,
        config=config,
        device=torch.device("cpu"),
    )

    assert sum(parameter.numel() for parameter in feature_mlp.parameters()) > sum(
        parameter.numel() for parameter in linear.parameters()
    )
