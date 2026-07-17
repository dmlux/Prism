from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PretrainedBackboneSpec:
    model_id: str
    revision: str
    trust_remote_code: bool

    def __post_init__(self) -> None:
        if not self.model_id or self.model_id.strip() != self.model_id:
            raise ValueError(
                "Backbone model ID must be non-empty "
                "and have no surrounding whitespace."
            )
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError(
                "Backbone revision must be a full lowercase Git commit hash."
            )
