"""PrismBERT model configuration — sizes chosen for the < 100 MB fp32 budget.

The backbone is the LTG GPT-BERT architecture (DeBERTa-style disentangled
relative positions, full bidirectional attention, GeGLU, parameter-free
pre-LayerNorm) — quant-friendly by construction and int8-fast on XNNPACK (the
NorBERT4 family). Dims are picked so the WHOLE shipped tagger (backbone + task
heads + character CNN + structured-morphology + lemma head) lands near, and
under, 100 MB in fp32.

Measured full-tagger sizes (English schema: 18 UPOS / 21 morphology features /
~1.6k lemma rules; production head WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_
CHARACTER_CNN):

    H320 · 12L · FF1024 · V16384 -> backbone 23.2M + heads 1.7M = 24.9M  ~99.6 MB  <- default (deeper: morphology/syntax + NorBERT4-family alignment)
    H384 · 8L · FF1024 · V16384 -> backbone 21.7M + heads 2.3M = 23.9M  ~95.7 MB  (wider fallback if UFeats disappoints)
    H384 · 10L · FF1024 · V16384 -> 27.8M                              ~111  MB (over budget)

Special-token convention mirrors the canonical gpt_bert tokenizer:
``<unk>``=0, ``<s>``=1 (cls/bos), ``</s>``=2 (sep/eos), ``<pad>``=3, ``<mask>``=4.

The modeling + configuration code is vendored in this package
(``modeling_gpt_bert.py``, ``configuration_gpt_bert.py``, from the Apache-2.0
BabyLM-community gpt_bert, adapted for transformers 5.x) — no
``trust_remote_code`` / third-party remote-code dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

# Special tokens (ids fixed to the gpt_bert convention; the trained tokenizer
# must place them at exactly these ids).
SPECIAL_TOKENS: dict[str, tuple[str, int]] = {
    "unk": ("<unk>", 0),
    "cls": ("<s>", 1),
    "sep": ("</s>", 2),
    "pad": ("<pad>", 3),
    "mask": ("<mask>", 4),
}

# Approximate full-tagger fp32 budget ceiling (MB); the resolver checks against it.
TAGGER_FP32_BUDGET_MB = 100.0


@dataclass(frozen=True, slots=True, kw_only=True)
class PrismBertConfig:
    """Backbone dimensions for a per-language PrismBERT."""

    hidden_size: int = 320
    num_layers: int = 12
    num_attention_heads: int = 5
    intermediate_size: int = 1024
    vocab_size: int = 16384
    position_bucket_size: int = 32
    max_position_embeddings: int = 512

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")


# The shipped English default (measured ~99.6 MB full tagger): deeper/narrower
# for morphology + syntax and alignment with the deep NorBERT4 family the Prism
# pipeline (pooling, LEARNED_LAST_FOUR aggregation, heads) is tuned on.
PRISM_BERT_EN = PrismBertConfig(
    hidden_size=320,
    num_layers=12,
    num_attention_heads=5,
    intermediate_size=1024,
    vocab_size=16384,
)


def build_gpt_bert_config(config: PrismBertConfig):
    """Return the vendored ``ModelConfig`` for the gpt_bert architecture."""

    from prism.prismbert.configuration_gpt_bert import ModelConfig

    return ModelConfig(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        position_bucket_size=config.position_bucket_size,
        max_position_embeddings=config.max_position_embeddings,
        pad_token_id=SPECIAL_TOKENS["pad"][1],
        bos_token_id=SPECIAL_TOKENS["cls"][1],
        eos_token_id=SPECIAL_TOKENS["sep"][1],
    )
