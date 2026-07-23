"""Build morphology-bundle reranker specifications from training targets."""

from collections import Counter
from collections.abc import Iterable, Mapping

from prism.data import TokenTargets
from prism.modeling.morphology_bundle_reranker import (
    MorphologyBundleCandidate,
    MorphologyBundleRerankerSpec,
)


def build_morphology_bundle_reranker_spec(
    *,
    targets: Iterable[TokenTargets],
    maximum_candidates_per_upos: int,
) -> MorphologyBundleRerankerSpec:
    if maximum_candidates_per_upos <= 0:
        raise ValueError("Maximum candidates per UPOS must be positive.")

    counts_by_upos: dict[
        int,
        Counter[tuple[tuple[bool, ...], ...]],
    ] = {}
    for target in targets:
        counts_by_upos.setdefault(target.upos_id, Counter())[target.morphology] += 1
    if not counts_by_upos:
        raise ValueError("Bundle reranker requires training targets.")

    candidates: list[MorphologyBundleCandidate] = []
    for upos_id, counts in sorted(counts_by_upos.items()):
        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:maximum_candidates_per_upos]
        candidates.extend(
            MorphologyBundleCandidate(
                upos_id=upos_id,
                morphology=morphology,
                training_count=count,
            )
            for morphology, count in ranked
        )

    return MorphologyBundleRerankerSpec(
        maximum_candidates_per_upos=maximum_candidates_per_upos,
        candidates=tuple(candidates),
    )


def serialize_morphology_bundle_reranker_spec(
    spec: MorphologyBundleRerankerSpec,
) -> dict[str, object]:
    return {
        "maximum_candidates_per_upos": spec.maximum_candidates_per_upos,
        "candidates": [
            {
                "upos_id": candidate.upos_id,
                "morphology": [
                    list(feature_labels) for feature_labels in candidate.morphology
                ],
                "training_count": candidate.training_count,
            }
            for candidate in spec.candidates
        ],
    }


def deserialize_morphology_bundle_reranker_spec(
    value: object,
) -> MorphologyBundleRerankerSpec:
    if not isinstance(value, Mapping):
        raise ValueError("Bundle-reranker checkpoint metadata must be a mapping.")
    maximum = value.get("maximum_candidates_per_upos")
    raw_candidates = value.get("candidates")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum <= 0
        or not isinstance(raw_candidates, (list, tuple))
    ):
        raise ValueError("Bundle-reranker checkpoint metadata is invalid.")

    candidates: list[MorphologyBundleCandidate] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("Bundle-reranker candidate metadata is invalid.")
        upos_id = raw_candidate.get("upos_id")
        training_count = raw_candidate.get("training_count")
        raw_morphology = raw_candidate.get("morphology")
        if (
            not isinstance(upos_id, int)
            or isinstance(upos_id, bool)
            or not isinstance(training_count, int)
            or isinstance(training_count, bool)
            or not isinstance(raw_morphology, (list, tuple))
        ):
            raise ValueError("Bundle-reranker candidate metadata is invalid.")
        morphology: list[tuple[bool, ...]] = []
        for raw_labels in raw_morphology:
            if not isinstance(raw_labels, (list, tuple)) or any(
                not isinstance(label, bool) for label in raw_labels
            ):
                raise ValueError("Bundle-reranker label metadata is invalid.")
            morphology.append(tuple(raw_labels))
        candidates.append(
            MorphologyBundleCandidate(
                upos_id=upos_id,
                morphology=tuple(morphology),
                training_count=training_count,
            )
        )

    return MorphologyBundleRerankerSpec(
        maximum_candidates_per_upos=maximum,
        candidates=tuple(candidates),
    )
