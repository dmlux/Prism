from prism.modeling.backbones import PretrainedBackboneSpec

NORBERT4_BASE_BACKBONE = PretrainedBackboneSpec(
    model_id="ltg/norbert4-base",
    revision="386ba2dc5ae5f95fec86d580c5fc4af34d380126",
    trust_remote_code=True,
    reinitialize_non_persistent_buffers=True,
)

NORBERT4_LARGE_BACKBONE = PretrainedBackboneSpec(
    model_id="ltg/norbert4-large",
    revision="49475ca0e59cc5db6ef2c762384b2a916ca8ead0",
    trust_remote_code=True,
    reinitialize_non_persistent_buffers=True,
)

NORBERT4_XSMALL_BACKBONE = PretrainedBackboneSpec(
    model_id="ltg/norbert4-xsmall",
    revision="7483327d36a2daa5dbe936c68aa277149c6f9632",
    trust_remote_code=True,
    reinitialize_non_persistent_buffers=True,
)
