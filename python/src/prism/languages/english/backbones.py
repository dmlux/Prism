"""Pretrained backbones for the English Prism profile.

The English profile keeps Prism's architecture and training pipeline
unchanged and only swaps the backbone family. NorBERT4 is the GPT-BERT
architecture; the closest modern English family that natively covers both the
~17M student size (≈ ``norbert4-xsmall``) and a large distillation teacher
(≈ ``norbert4-large``) is the Ettin encoder suite (ModernBERT lineage,
Weller et al., 2025), trained on 2T open tokens under the MIT license.

Unlike NorBERT4, Ettin/ModernBERT is a first-class ``transformers``
architecture, so ``trust_remote_code`` is not required. Two settings are
pinned for a clean, deterministic ExecuTorch export (verified by the export
spike, parity 2.7e-5 through the existing XNNPACK lowering path):

* ``attention_implementation="eager"`` — avoids the SDPA/flash and unpadding
  paths so ``torch.export(strict=True)`` captures a portable attention graph;
* ``reference_compile=False`` — disables ModernBERT's internal ``torch.compile``
  wrapping, which must not run during export capture or MPS training.
"""

from prism.modeling.backbones import PretrainedBackboneSpec

# Student (ships): 16.80M parameters, ≈ ltg/norbert4-xsmall (17M).
ETTIN_ENCODER_17M_BACKBONE = PretrainedBackboneSpec(
    model_id="jhu-clsp/ettin-encoder-17m",
    revision="987607455c61e7a5bbc85f7758e0512ea6d0ae4c",
    trust_remote_code=False,
    attention_implementation="eager",
    config_overrides=(("reference_compile", False),),
)

# Teacher (distillation, development-only): ≈ 400M, ≈ ltg/norbert4-large (360M).
# The teacher is never exported, so it keeps the training-time attention
# default; reference_compile stays off for deterministic MPS runs.
ETTIN_ENCODER_400M_BACKBONE = PretrainedBackboneSpec(
    model_id="jhu-clsp/ettin-encoder-400m",
    revision="7662476d60abb071a5bd319c9f3074f3072c062d",
    trust_remote_code=False,
    config_overrides=(("reference_compile", False),),
)
