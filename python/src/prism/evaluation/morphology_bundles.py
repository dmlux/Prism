"""Training-derived candidate inventories for complete morphology bundles."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


MorphologyBundle = tuple[str, ...]


def morphology_bundle_from_features(
    features: Mapping[str, str],
) -> MorphologyBundle:
    return tuple(sorted(f"{name}={value}" for name, value in features.items()))


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleExample:
    upos: str
    bundle: MorphologyBundle

    def __post_init__(self) -> None:
        if not self.upos or self.upos.strip() != self.upos:
            raise ValueError("Morphology-bundle UPOS must be non-empty and trimmed.")
        if self.bundle != tuple(sorted(self.bundle)):
            raise ValueError("Morphology bundle values must be sorted.")


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleCandidateSet:
    upos: str
    bundles: tuple[MorphologyBundle, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleInventory:
    training_example_count: int
    candidate_sets: tuple[MorphologyBundleCandidateSet, ...]

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[MorphologyBundleExample],
    ) -> "MorphologyBundleInventory":
        counts_by_upos: dict[str, Counter[MorphologyBundle]] = {}
        example_count = 0
        for example in examples:
            counts_by_upos.setdefault(example.upos, Counter())[example.bundle] += 1
            example_count += 1

        if not example_count:
            raise ValueError("Morphology-bundle inventory requires training examples.")

        return cls(
            training_example_count=example_count,
            candidate_sets=tuple(
                MorphologyBundleCandidateSet(
                    upos=upos,
                    bundles=tuple(
                        bundle
                        for bundle, _ in sorted(
                            bundle_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ),
                )
                for upos, bundle_counts in sorted(counts_by_upos.items())
            ),
        )

    @property
    def distinct_bundle_count(self) -> int:
        return len(
            {
                bundle
                for candidate_set in self.candidate_sets
                for bundle in candidate_set.bundles
            }
        )

    @property
    def distinct_upos_bundle_count(self) -> int:
        return sum(len(candidate_set.bundles) for candidate_set in self.candidate_sets)

    @property
    def maximum_candidate_count(self) -> int:
        return max(len(candidate_set.bundles) for candidate_set in self.candidate_sets)

    def candidates_for(self, upos: str) -> tuple[MorphologyBundle, ...]:
        for candidate_set in self.candidate_sets:
            if candidate_set.upos == upos:
                return candidate_set.bundles
        return ()


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleTopKCoverage:
    candidate_count: int
    covered_token_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleCoverage:
    token_count: int
    globally_seen_token_count: int
    gold_upos_seen_token_count: int
    top_k: tuple[MorphologyBundleTopKCoverage, ...]

    @property
    def global_coverage(self) -> float:
        return self.globally_seen_token_count / self.token_count

    @property
    def gold_upos_coverage(self) -> float:
        return self.gold_upos_seen_token_count / self.token_count


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyBundleOracleMetrics:
    training_example_count: int
    distinct_bundle_count: int
    distinct_upos_bundle_count: int
    maximum_candidate_count: int
    overall: MorphologyBundleCoverage
    annotated: MorphologyBundleCoverage


def evaluate_morphology_bundle_oracle(
    *,
    inventory: MorphologyBundleInventory,
    development_examples: Sequence[MorphologyBundleExample],
    candidate_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> MorphologyBundleOracleMetrics:
    if not development_examples:
        raise ValueError("Morphology-bundle oracle requires Development examples.")
    if (
        not candidate_counts
        or any(count <= 0 for count in candidate_counts)
        or candidate_counts != tuple(sorted(set(candidate_counts)))
    ):
        raise ValueError("Candidate counts must be unique, sorted, and positive.")

    global_bundles = {
        bundle
        for candidate_set in inventory.candidate_sets
        for bundle in candidate_set.bundles
    }

    def coverage(
        examples: Sequence[MorphologyBundleExample],
    ) -> MorphologyBundleCoverage:
        if not examples:
            raise ValueError("Morphology-bundle coverage slice must not be empty.")

        global_seen_count = 0
        gold_upos_seen_count = 0
        top_k_counts = [0] * len(candidate_counts)
        for example in examples:
            candidates = inventory.candidates_for(example.upos)
            global_seen_count += example.bundle in global_bundles
            gold_upos_seen_count += example.bundle in candidates
            for index, candidate_count in enumerate(candidate_counts):
                top_k_counts[index] += example.bundle in candidates[:candidate_count]

        return MorphologyBundleCoverage(
            token_count=len(examples),
            globally_seen_token_count=global_seen_count,
            gold_upos_seen_token_count=gold_upos_seen_count,
            top_k=tuple(
                MorphologyBundleTopKCoverage(
                    candidate_count=candidate_count,
                    covered_token_count=covered_count,
                )
                for candidate_count, covered_count in zip(
                    candidate_counts,
                    top_k_counts,
                    strict=True,
                )
            ),
        )

    annotated_examples = tuple(
        example for example in development_examples if example.bundle
    )
    return MorphologyBundleOracleMetrics(
        training_example_count=inventory.training_example_count,
        distinct_bundle_count=inventory.distinct_bundle_count,
        distinct_upos_bundle_count=inventory.distinct_upos_bundle_count,
        maximum_candidate_count=inventory.maximum_candidate_count,
        overall=coverage(development_examples),
        annotated=coverage(annotated_examples),
    )
