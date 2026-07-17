from dataclasses import dataclass

from transformers import AutoModel, PreTrainedModel


@dataclass(frozen=True, slots=True, kw_only=True)
class PretrainedBackboneSpec:
    model_id: str
    revision: str
    trust_remote_code: bool
    reinitialize_non_persistent_buffers: bool = False

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


def _reinitialize_non_persistent_buffers(model: PreTrainedModel) -> PreTrainedModel:
    reinitialized_model = type(model)(model.config)
    reinitialized_model.load_state_dict(model.state_dict(), strict=True)
    reinitialized_model.train(model.training)
    return reinitialized_model


def load_backbone_model(
    spec: PretrainedBackboneSpec,
) -> PreTrainedModel:
    from_pretrained = AutoModel.from_pretrained
    model = from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )

    if model is None:
        raise RuntimeError("Backbone model could not be loaded.")

    if spec.reinitialize_non_persistent_buffers:
        model = _reinitialize_non_persistent_buffers(model)

    return model
