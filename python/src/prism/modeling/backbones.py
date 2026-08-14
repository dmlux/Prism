from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModel, PreTrainedModel


@dataclass(frozen=True, slots=True, kw_only=True)
class PretrainedBackboneSpec:
    model_id: str
    revision: str
    trust_remote_code: bool
    reinitialize_non_persistent_buffers: bool = False
    # Optional ``attn_implementation`` for ``from_pretrained`` (e.g. "eager",
    # required by ModernBERT for a portable ExecuTorch export graph). ``None``
    # keeps the transformers default.
    attention_implementation: str | None = None
    # Config attributes to set after loading, as ordered ``(name, value)`` pairs
    # (e.g. ``("reference_compile", False)`` for ModernBERT). Empty leaves the
    # loaded config untouched, preserving the prior behaviour.
    config_overrides: tuple[tuple[str, object], ...] = ()
    # Per-backbone int8 discriminator override; None uses the profile's.
    quantization: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or self.model_id.strip() != self.model_id:
            raise ValueError(
                "Backbone model ID must be non-empty "
                "and have no surrounding whitespace."
            )
        if self.revision == "local":
            if not Path(self.model_id).is_dir():
                raise ValueError(
                    "A 'local' backbone revision requires model_id to be an "
                    f"existing directory; {self.model_id!r} is not one."
                )
            return
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError(
                "Backbone revision must be a full lowercase Git commit hash "
                "(or 'local')."
            )


def _reinitialize_non_persistent_buffers(model: PreTrainedModel) -> PreTrainedModel:
    reinitialized_model = type(model)(model.config)
    reinitialized_model.load_state_dict(model.state_dict(), strict=True)
    reinitialized_model.train(model.training)
    return reinitialized_model


def load_backbone_model(
    spec: PretrainedBackboneSpec,
) -> PreTrainedModel:
    load_keyword_arguments: dict[str, object] = {
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.revision != "local":
        load_keyword_arguments["revision"] = spec.revision
    if spec.attention_implementation is not None:
        load_keyword_arguments["attn_implementation"] = spec.attention_implementation

    model = AutoModel.from_pretrained(spec.model_id, **load_keyword_arguments)

    if model is None:
        raise RuntimeError("Backbone model could not be loaded.")

    for attribute_name, attribute_value in spec.config_overrides:
        setattr(model.config, attribute_name, attribute_value)

    if spec.reinitialize_non_persistent_buffers:
        model = _reinitialize_non_persistent_buffers(model)

    return model
