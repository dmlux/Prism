from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


UPOS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class UposSchema:
    version: int
    labels: tuple[str, ...]
    _label_ids: Mapping[str, int] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("UPOS schema version must be positive.")
        if not self.labels:
            raise ValueError("UPOS schema must contain labels.")
        if any(not label or label.strip() != label for label in self.labels):
            raise ValueError(
                "UPOS labels must be non-empty and have no surrounding whitespace."
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("UPOS labels must be unique.")
        if self.labels != tuple(sorted(self.labels)):
            raise ValueError("UPOS labels must be sorted.")

        object.__setattr__(
            self,
            "_label_ids",
            MappingProxyType(
                {label: label_id for label_id, label in enumerate(self.labels)}
            ),
        )

    def label_id_for(
        self,
        label: str,
    ) -> int | None:
        return self._label_ids.get(label)

    def label_for_id(
        self,
        label_id: int,
    ) -> str:
        if label_id < 0 or label_id >= len(self.labels):
            raise ValueError(f"UPOS label ID {label_id} is out of range.")

        return self.labels[label_id]
