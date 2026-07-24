from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math

import torch
from torch import Tensor, nn

from prism.modeling import (
    TaskResidualAdapter,
    TokenClassificationHead,
    TokenTagger,
    WideSharedResidualTokenProjection,
    apply_morphology_feature_logit_correction,
    decode_morphology_feature_logits,
)
from prism.schema import MorphologySchema
from prism.training.batches import SupervisedTokenTaskBatch
from prism.training.losses import calculate_morphology_feature_loss


class MorphologyHeadProbeArchitecture(StrEnum):
    LINEAR = "linear"
    SHARED_MLP = "shared-mlp"
    FEATURE_MLP = "feature-mlp"


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyHeadProbeConfig:
    epoch_count: int = 8
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    dropout_probability: float = 0.1
    max_gradient_norm: float = 1.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.epoch_count <= 0:
            raise ValueError("Probe epoch count must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Probe batch size must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Probe learning rate must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("Probe weight decay must be finite and non-negative.")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "Probe dropout probability must be at least zero and below one."
            )
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0:
            raise ValueError("Probe gradient norm must be finite and positive.")
        if self.random_seed < 0:
            raise ValueError("Probe random seed must be non-negative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyProbeSlice:
    name: str
    mask: Tensor

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("Probe slice name must be non-empty and trimmed.")
        if self.mask.ndim != 1 or self.mask.dtype != torch.bool:
            raise ValueError("Probe slice mask must be a one-dimensional bool tensor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyProbeDataset:
    hidden_states: Tensor
    morphology_targets: tuple[Tensor, ...]
    slices: tuple[MorphologyProbeSlice, ...] = ()

    def __post_init__(self) -> None:
        if self.hidden_states.ndim != 2:
            raise ValueError("Probe hidden states must have two dimensions.")
        if not self.hidden_states.is_floating_point():
            raise ValueError("Probe hidden states must be floating point.")
        if self.hidden_states.shape[0] <= 0 or self.hidden_states.shape[1] <= 0:
            raise ValueError("Probe hidden-state dimensions must be positive.")
        if not self.morphology_targets:
            raise ValueError("Probe dataset must contain morphology targets.")
        if any(
            targets.ndim != 2
            or targets.shape[0] != self.hidden_states.shape[0]
            or targets.dtype != torch.bool
            for targets in self.morphology_targets
        ):
            raise ValueError(
                "Probe morphology targets must be two-dimensional bool tensors "
                "with the same token count as hidden states."
            )
        if len({token_slice.name for token_slice in self.slices}) != len(self.slices):
            raise ValueError("Probe slice names must be unique.")
        if any(
            token_slice.mask.shape != (self.hidden_states.shape[0],)
            for token_slice in self.slices
        ):
            raise ValueError("Probe slice masks must match the dataset token count.")

    @property
    def token_count(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.hidden_states.shape[1]


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyProbeAccuracy:
    correct_count: int
    token_count: int

    def __post_init__(self) -> None:
        if self.correct_count < 0 or self.token_count < 0:
            raise ValueError("Probe metric counts must be non-negative.")
        if self.correct_count > self.token_count:
            raise ValueError("Probe correct count cannot exceed token count.")

    @property
    def accuracy(self) -> float | None:
        if self.token_count == 0:
            return None
        return self.correct_count / self.token_count


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyProbeSliceMetrics:
    name: str
    accuracy: MorphologyProbeAccuracy


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyProbeFeatureMetrics:
    feature_name: str
    overall: MorphologyProbeAccuracy
    annotated: MorphologyProbeAccuracy
    slices: tuple[MorphologyProbeSliceMetrics, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyHeadProbeResult:
    architecture: MorphologyHeadProbeArchitecture
    parameter_count: int
    epoch_losses: tuple[float, ...]
    features: tuple[MorphologyProbeFeatureMetrics, ...]


class MorphologyHeadProbe(nn.Module):
    """Small diagnostic heads trained on a frozen token representation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        morphology_schema: MorphologySchema,
        architecture: MorphologyHeadProbeArchitecture,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("Probe hidden size must be positive.")

        self.architecture = architecture
        self.shared_projection: nn.Module
        if architecture is MorphologyHeadProbeArchitecture.SHARED_MLP:
            self.shared_projection = WideSharedResidualTokenProjection(
                hidden_size=hidden_size,
                dropout_probability=dropout_probability,
            )
        else:
            self.shared_projection = nn.Identity()

        if architecture is MorphologyHeadProbeArchitecture.FEATURE_MLP:
            bottleneck_size = max(1, hidden_size // 2)
            self.feature_heads = nn.ModuleList(
                nn.Sequential(
                    TaskResidualAdapter(
                        hidden_size=hidden_size,
                        bottleneck_size=bottleneck_size,
                        dropout_probability=dropout_probability,
                    ),
                    TokenClassificationHead(
                        hidden_size=hidden_size,
                        label_count=feature.logit_count,
                        dropout_probability=dropout_probability,
                    ),
                )
                for feature in morphology_schema.features
            )
        elif architecture in (
            MorphologyHeadProbeArchitecture.LINEAR,
            MorphologyHeadProbeArchitecture.SHARED_MLP,
        ):
            self.feature_heads = nn.ModuleList(
                TokenClassificationHead(
                    hidden_size=hidden_size,
                    label_count=feature.logit_count,
                    dropout_probability=dropout_probability,
                )
                for feature in morphology_schema.features
            )
        else:
            raise ValueError(f"Unsupported morphology probe: {architecture!r}")

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, ...]:
        projected_hidden_states = self.shared_projection(hidden_states)
        return tuple(head(projected_hidden_states) for head in self.feature_heads)


def extract_morphology_probe_dataset(
    *,
    model: TokenTagger,
    batches: Iterable[SupervisedTokenTaskBatch],
    device: torch.device,
    batch_slice_masks: Mapping[str, Sequence[Tensor]] | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> MorphologyProbeDataset:
    """Run a frozen tagger once and retain only valid morphology representations."""

    model.requires_grad_(False)
    model.eval()
    hidden_state_batches: list[Tensor] = []
    target_batches: list[list[Tensor]] | None = None
    slice_batches = {
        name: [] for name in (() if batch_slice_masks is None else batch_slice_masks)
    }
    batch_count = 0

    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(batches):
            batch_count += 1
            if on_batch is not None:
                on_batch(batch_count)
            batch = cpu_batch.to(device)
            task_hidden_states = model.encode_task_hidden_states(
                batch.model_inputs,
                character_batch=batch.character_inputs,
            )
            token_mask = batch.targets.token_mask
            hidden_state_batches.append(
                task_hidden_states.morphology[token_mask].to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )

            if target_batches is None:
                target_batches = [[] for _ in batch.targets.morphology_targets]
            for feature_batches, feature_targets in zip(
                target_batches,
                batch.targets.morphology_targets,
                strict=True,
            ):
                feature_batches.append(feature_targets[token_mask].cpu())

            if batch_slice_masks is not None:
                for name, masks in batch_slice_masks.items():
                    if batch_index >= len(masks):
                        raise ValueError(
                            f"Probe slice {name!r} has too few batch masks."
                        )
                    slice_mask = masks[batch_index]
                    if slice_mask.shape != cpu_batch.targets.token_mask.shape:
                        raise ValueError(
                            f"Probe slice {name!r} does not match its batch."
                        )
                    slice_batches[name].append(
                        slice_mask[cpu_batch.targets.token_mask].cpu()
                    )

    if batch_count == 0 or target_batches is None:
        raise ValueError("Probe extraction requires at least one batch.")
    if batch_slice_masks is not None and any(
        len(masks) != batch_count for masks in batch_slice_masks.values()
    ):
        raise ValueError("Probe slice masks must match the extracted batch count.")

    return MorphologyProbeDataset(
        hidden_states=torch.cat(hidden_state_batches),
        morphology_targets=tuple(
            torch.cat(feature_batches) for feature_batches in target_batches
        ),
        slices=tuple(
            MorphologyProbeSlice(
                name=name,
                mask=torch.cat(masks),
            )
            for name, masks in slice_batches.items()
        ),
    )


def train_morphology_head_probe(
    *,
    dataset: MorphologyProbeDataset,
    morphology_schema: MorphologySchema,
    architecture: MorphologyHeadProbeArchitecture,
    config: MorphologyHeadProbeConfig,
    device: torch.device,
    morphology_weights: tuple[Tensor, ...] | None = None,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[MorphologyHeadProbe, tuple[float, ...]]:
    if len(dataset.morphology_targets) != len(morphology_schema.features):
        raise ValueError("Probe dataset must match the morphology schema.")
    if morphology_weights is not None and len(morphology_weights) != len(
        morphology_schema.features
    ):
        raise ValueError("Probe morphology weights must match the schema.")

    torch.manual_seed(config.random_seed)
    probe = MorphologyHeadProbe(
        hidden_size=dataset.hidden_size,
        morphology_schema=morphology_schema,
        architecture=architecture,
        dropout_probability=config.dropout_probability,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resolved_weights = (
        (None,) * len(morphology_schema.features)
        if morphology_weights is None
        else tuple(weights.to(device) for weights in morphology_weights)
    )
    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    epoch_losses: list[float] = []

    for epoch_index in range(config.epoch_count):
        probe.train()
        permutation = torch.randperm(
            dataset.token_count,
            generator=generator,
        )
        epoch_loss_sum = 0.0
        epoch_token_count = 0

        for start in range(0, dataset.token_count, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            hidden_states = dataset.hidden_states[indices].to(device)
            targets = tuple(
                feature_targets[indices].to(device)
                for feature_targets in dataset.morphology_targets
            )
            logits = probe(hidden_states)
            token_mask = torch.ones(
                hidden_states.shape[0],
                dtype=torch.bool,
                device=device,
            )
            feature_losses = tuple(
                calculate_morphology_feature_loss(
                    feature_logits=feature_logits,
                    feature_targets=feature_targets,
                    feature_schema=feature_schema,
                    token_mask=token_mask,
                    feature_weights=feature_weights,
                )
                for (
                    feature_logits,
                    feature_targets,
                    feature_schema,
                    feature_weights,
                ) in zip(
                    logits,
                    targets,
                    morphology_schema.features,
                    resolved_weights,
                    strict=True,
                )
            )
            loss = torch.stack(feature_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                probe.parameters(),
                max_norm=config.max_gradient_norm,
            )
            optimizer.step()

            batch_token_count = hidden_states.shape[0]
            epoch_loss_sum += float(loss.detach().cpu()) * batch_token_count
            epoch_token_count += batch_token_count

        epoch_loss = epoch_loss_sum / epoch_token_count
        epoch_losses.append(epoch_loss)
        if on_epoch is not None:
            on_epoch(epoch_index + 1, epoch_loss)

    return probe, tuple(epoch_losses)


def _accuracy(
    *,
    correct: Tensor,
    mask: Tensor,
) -> MorphologyProbeAccuracy:
    return MorphologyProbeAccuracy(
        correct_count=int((correct & mask).sum().item()),
        token_count=int(mask.sum().item()),
    )


def evaluate_morphology_head_probe(
    *,
    probe: MorphologyHeadProbe,
    dataset: MorphologyProbeDataset,
    morphology_schema: MorphologySchema,
    device: torch.device,
    batch_size: int,
    morphology_weights: tuple[Tensor, ...] | None = None,
    logit_correction_strength: float = 0.0,
) -> tuple[MorphologyProbeFeatureMetrics, ...]:
    if batch_size <= 0:
        raise ValueError("Probe evaluation batch size must be positive.")
    if len(dataset.morphology_targets) != len(morphology_schema.features):
        raise ValueError("Probe dataset must match the morphology schema.")
    if not 0.0 <= logit_correction_strength <= 1.0:
        raise ValueError("Probe logit correction must be between zero and one.")
    if logit_correction_strength > 0.0 and morphology_weights is None:
        raise ValueError("Probe logit correction requires morphology weights.")
    if morphology_weights is not None and len(morphology_weights) != len(
        morphology_schema.features
    ):
        raise ValueError("Probe morphology weights must match the schema.")

    predictions: list[list[Tensor]] = [[] for _ in morphology_schema.features]
    probe.eval()
    with torch.no_grad():
        for start in range(0, dataset.token_count, batch_size):
            hidden_states = dataset.hidden_states[start : start + batch_size].to(device)
            logits = probe(hidden_states)
            for feature_index, (feature_logits, feature_schema) in enumerate(
                zip(logits, morphology_schema.features, strict=True)
            ):
                if morphology_weights is not None:
                    feature_logits = apply_morphology_feature_logit_correction(
                        feature_logits=feature_logits,
                        feature_schema=feature_schema,
                        weights=morphology_weights[feature_index],
                        strength=logit_correction_strength,
                    )
                predictions[feature_index].append(
                    decode_morphology_feature_logits(
                        feature_logits=feature_logits,
                        feature_schema=feature_schema,
                    ).cpu()
                )

    feature_metrics: list[MorphologyProbeFeatureMetrics] = []
    all_tokens = torch.ones(dataset.token_count, dtype=torch.bool)
    for feature_schema, feature_predictions, targets in zip(
        morphology_schema.features,
        predictions,
        dataset.morphology_targets,
        strict=True,
    ):
        resolved_predictions = torch.cat(feature_predictions)
        correct = (resolved_predictions == targets).all(dim=-1)
        annotated = ~targets[:, 0]
        feature_metrics.append(
            MorphologyProbeFeatureMetrics(
                feature_name=feature_schema.name,
                overall=_accuracy(correct=correct, mask=all_tokens),
                annotated=_accuracy(correct=correct, mask=annotated),
                slices=tuple(
                    MorphologyProbeSliceMetrics(
                        name=token_slice.name,
                        accuracy=_accuracy(
                            correct=correct,
                            mask=token_slice.mask,
                        ),
                    )
                    for token_slice in dataset.slices
                ),
            )
        )

    return tuple(feature_metrics)


def run_morphology_head_probe(
    *,
    training_dataset: MorphologyProbeDataset,
    evaluation_dataset: MorphologyProbeDataset,
    morphology_schema: MorphologySchema,
    architecture: MorphologyHeadProbeArchitecture,
    config: MorphologyHeadProbeConfig,
    device: torch.device,
    morphology_weights: tuple[Tensor, ...] | None = None,
    logit_correction_strength: float = 0.0,
    on_epoch: Callable[[int, float], None] | None = None,
) -> MorphologyHeadProbeResult:
    probe, epoch_losses = train_morphology_head_probe(
        dataset=training_dataset,
        morphology_schema=morphology_schema,
        architecture=architecture,
        config=config,
        device=device,
        morphology_weights=morphology_weights,
        on_epoch=on_epoch,
    )
    return MorphologyHeadProbeResult(
        architecture=architecture,
        parameter_count=sum(parameter.numel() for parameter in probe.parameters()),
        epoch_losses=epoch_losses,
        features=evaluate_morphology_head_probe(
            probe=probe,
            dataset=evaluation_dataset,
            morphology_schema=morphology_schema,
            device=device,
            batch_size=config.batch_size,
            morphology_weights=morphology_weights,
            logit_correction_strength=logit_correction_strength,
        ),
    )


def serialize_morphology_probe_dataset(
    dataset: MorphologyProbeDataset,
) -> dict[str, object]:
    return {
        "hidden_states": dataset.hidden_states,
        "morphology_targets": dataset.morphology_targets,
        "slices": {
            token_slice.name: token_slice.mask for token_slice in dataset.slices
        },
    }


def deserialize_morphology_probe_dataset(
    raw_dataset: Mapping[str, object],
) -> MorphologyProbeDataset:
    hidden_states = raw_dataset.get("hidden_states")
    morphology_targets = raw_dataset.get("morphology_targets")
    raw_slices = raw_dataset.get("slices", {})
    if not isinstance(hidden_states, Tensor):
        raise ValueError("Probe cache hidden states are invalid.")
    if not isinstance(morphology_targets, (list, tuple)) or not all(
        isinstance(targets, Tensor) for targets in morphology_targets
    ):
        raise ValueError("Probe cache morphology targets are invalid.")
    if not isinstance(raw_slices, Mapping) or not all(
        isinstance(name, str) and isinstance(mask, Tensor)
        for name, mask in raw_slices.items()
    ):
        raise ValueError("Probe cache slices are invalid.")
    return MorphologyProbeDataset(
        hidden_states=hidden_states,
        morphology_targets=tuple(morphology_targets),
        slices=tuple(
            MorphologyProbeSlice(name=name, mask=mask)
            for name, mask in raw_slices.items()
        ),
    )
