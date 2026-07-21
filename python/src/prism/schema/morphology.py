from dataclasses import dataclass


NO_MORPHOLOGY_VALUE = "<NONE>"
MORPHOLOGY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologyFeatureSchema:
    name: str
    values: tuple[str, ...]
    allows_multiple_values: bool

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError(
                "Morphology feature name must be non-empty "
                "and must not have surrounding whitespace."
            )

        if not self.values:
            raise ValueError("Morphology feature values must not be empty.")

        if NO_MORPHOLOGY_VALUE in self.values:
            raise ValueError(
                f"{NO_MORPHOLOGY_VALUE} is reserved for missing annotations."
            )

        if len(set(self.values)) != len(self.values):
            raise ValueError("Morphology feature values must be unique.")

        if self.values != tuple(sorted(self.values)):
            raise ValueError("Morphology feature values must be sorted.")

    @property
    def labels(self) -> tuple[str, ...]:
        return (NO_MORPHOLOGY_VALUE, *self.values)

    @property
    def logit_count(self) -> int:
        if self.allows_multiple_values:
            return len(self.values)

        return len(self.labels)


@dataclass(frozen=True, slots=True, kw_only=True)
class MorphologySchema:
    version: int
    features: tuple[MorphologyFeatureSchema, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Morphology schema version must be positive.")

        if not self.features:
            raise ValueError("Morphology schema features must not be empty.")

        feature_names = tuple(feature.name for feature in self.features)

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Morphology feature names must be unique.")

        if feature_names != tuple(sorted(feature_names)):
            raise ValueError("Morphology features must be sorted by name.")
