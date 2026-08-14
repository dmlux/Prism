"""Register the vendored gpt_bert architecture with transformers' Auto classes,
so a locally-saved PrismBERT backbone (config model_type='gpt_bert', saved as
GPTBERTForMaskedLM) loads as its base encoder via AutoModel.from_pretrained.

Idempotent: safe to call many times.
"""
from __future__ import annotations

_registered = False


def register_gpt_bert() -> None:
    global _registered
    if _registered:
        return
    from transformers import AutoConfig, AutoModel

    from prism.bert.configuration_gpt_bert import ModelConfig
    from prism.bert.modeling_gpt_bert import GPTBERT

    try:
        AutoConfig.register("gpt_bert", ModelConfig)
    except ValueError:
        pass  # already registered in this interpreter
    try:
        AutoModel.register(ModelConfig, GPTBERT)
    except ValueError:
        pass
    _registered = True
