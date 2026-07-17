from prism.modeling.backbones import PretrainedBackboneSpec


NORBERT4_XSMALL_BACKBONE = PretrainedBackboneSpec(
    model_id="ltg/norbert4-xsmall",
    revision="7483327d36a2daa5dbe936c68aa277149c6f9632",
    trust_remote_code=True,
    reinitialize_non_persistent_buffers=True,
)
