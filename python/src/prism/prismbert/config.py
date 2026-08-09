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

NOTE (pre-ship follow-up): the config/model code is currently obtained from the
canonical ``gpt_bert`` reference via ``trust_remote_code``; vendor the modeling
+ configuration modules into this package (transformers-5.x compatible) before
publishing PrismBERT so we own the code and drop the third-party remote-code
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

# The canonical gpt_bert reference we template the config from (until vendored).
GPT_BERT_REFERENCE = "BabyLM-community/babylm-baseline-100m-gpt-bert-mixed"
GPT_BERT_REFERENCE_REVISION = "09629ffe557c4143aa7b857f92004f3e45689eff"

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
    """Return a ``transformers`` config object for the gpt_bert architecture.

    Templates the canonical gpt_bert config (via ``trust_remote_code``) and
    overrides the dimensions + special-token ids. Requires the optional
    ``datasets``/``transformers`` remote-code fetch on first use.
    """

    import transformers

    # transformers 5.x reads ``all_tied_weights_keys`` during load; the canonical
    # gpt_bert code predates it. Harmless empty-dict fallback.
    pretrained_model = transformers.modeling_utils.PreTrainedModel
    if not isinstance(getattr(pretrained_model, "all_tied_weights_keys", None), dict):
        pretrained_model.all_tied_weights_keys = {}

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        GPT_BERT_REFERENCE,
        revision=GPT_BERT_REFERENCE_REVISION,
        trust_remote_code=True,
    )
    hf_config.hidden_size = config.hidden_size
    hf_config.num_layers = config.num_layers
    hf_config.num_hidden_layers = config.num_layers
    hf_config.num_attention_heads = config.num_attention_heads
    hf_config.intermediate_size = config.intermediate_size
    hf_config.vocab_size = config.vocab_size
    hf_config.position_bucket_size = config.position_bucket_size
    hf_config.max_position_embeddings = config.max_position_embeddings
    for role, (_, token_id) in SPECIAL_TOKENS.items():
        setattr(hf_config, f"{role}_token_id", token_id)
    return hf_config
