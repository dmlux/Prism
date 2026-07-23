"""Checkpoint serialization for local morphology agreement refinement."""

from collections.abc import Mapping

from prism.modeling.morphology_agreement import MorphologyAgreementRefinerSpec


def serialize_morphology_agreement_refiner_spec(
    spec: MorphologyAgreementRefinerSpec,
) -> dict[str, object]:
    return {
        "window_radius": spec.window_radius,
        "bottleneck_size": spec.bottleneck_size,
        "target_feature_names": list(spec.target_feature_names),
    }


def deserialize_morphology_agreement_refiner_spec(
    value: object,
) -> MorphologyAgreementRefinerSpec:
    if not isinstance(value, Mapping):
        raise ValueError("Morphology agreement metadata must be a mapping.")

    window_radius = value.get("window_radius")
    bottleneck_size = value.get("bottleneck_size")
    raw_target_feature_names = value.get("target_feature_names")
    if (
        not isinstance(window_radius, int)
        or isinstance(window_radius, bool)
        or not isinstance(bottleneck_size, int)
        or isinstance(bottleneck_size, bool)
        or not isinstance(raw_target_feature_names, (list, tuple))
        or any(not isinstance(name, str) for name in raw_target_feature_names)
    ):
        raise ValueError("Morphology agreement metadata is invalid.")

    return MorphologyAgreementRefinerSpec(
        window_radius=window_radius,
        bottleneck_size=bottleneck_size,
        target_feature_names=tuple(raw_target_feature_names),
    )
