"""LTG's GPT-BERT architecture with RoPE — Prism's default per-language,
quant-friendly backbone.

Prism-authored adapter over the vendored modeling
(``modeling_gpt_bert_rope.py`` + ``configuration_gpt_bert_rope.py``, from
``ltg/norbert4-base``, Apache-2.0, vendored + modified — that HF model is where
this architecture is published). This module is what the rest of Prism uses: a
budget-sized config and a STANDARD bidirectional masked-LM head/forward, so the
architecture trains inside Prism's existing MLM pretraining pipeline (15 %
masking, HF ``Trainer``, corpus interleaving, resume-safe) and loads as a plain
encoder for the tagger.

The pretraining OBJECTIVE is standard bidirectional BERT-MLM — how the tagger
consumes the backbone — NOT the vendored objective
(:class:`~prism.bert.modeling_gpt_bert_rope.GptBertForMaskedLM`, whose
``forward`` implements LTG's shifted causal/MLM recipe; kept untouched for
reference). The shipped products trained on this backbone are the per-language
PrismBERTs (``prismbert-en``, ``prismbert-de``, …).

Apache-2.0. The architecture instantiated here is the vendored GPT-BERT code
(see those files' headers for attribution).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput

from prism.bert.configuration_gpt_bert_rope import GptBertConfig
from prism.bert.modeling_gpt_bert_rope import GptBertModel

_registered = False


class GptBertRopeConfig(GptBertConfig):
    """HF config for the RoPE GPT-BERT backbone — the vendored
    :class:`GptBertConfig` plus a stable ``model_type`` so a locally-saved
    backbone dispatches through transformers' Auto classes on reload (the
    vendored config leaves ``model_type`` unset)."""

    model_type = "gpt-bert-rope"


class GptBertRopeEncoder(GptBertModel):
    """The bidirectional encoder (no MLM head) — what the tagger loads via
    ``AutoModel.from_pretrained``. Identical to the vendored
    :class:`GptBertModel` but carries our ``config_class`` so
    ``AutoModel.register`` accepts it (transformers requires
    ``model_class.config_class`` to match the registered config)."""

    config_class = GptBertRopeConfig


class GptBertRopeForMaskedLM(GptBertModel):
    """RoPE GPT-BERT encoder + the vendored ``LMClassifier`` head, trained with a
    STANDARD (non-shifted, unbounded) masked-LM cross-entropy so it plugs into
    :class:`~transformers.DataCollatorForLanguageModeling`.

    Deliberately distinct from the vendored :class:`GptBertForMaskedLM`, whose
    ``forward`` implements LTG's shifted causal/MLM objective (``labels[:, 1:]``
    vs ``pred[:, :-1]`` + ``30*sigmoid`` bounding + BOS-prepend) — the GPT-BERT
    generative recipe, which we do not use for an encoder-only tagging backbone.
    """

    config_class = GptBertRopeConfig
    # transformers 5.x maps {tied output key: source key}; the MLM projection is
    # tied to the input word embedding.
    _tied_weights_keys = {"classifier.emb2vocab.weight": "embedding.word_embedding.weight"}

    def __init__(self, config: GptBertConfig, **kwargs):
        # Bidirectional encoder (symmetric attention window). Set before the
        # nn.Module __init__ runs — a plain bool, safe pre-super, and GptBertModel
        # reads self.is_decoder if already present.
        self.is_decoder = False
        super().__init__(config, add_mlm_layer=True, **kwargs)
        # Weight-tie the MLM projection to the input embedding (as the vendored
        # model does): share the Parameter object so the tie holds through
        # training + saving.
        self.classifier.emb2vocab.weight = self.embedding.word_embedding.weight

    def get_output_embeddings(self):
        return self.classifier.emb2vocab

    def set_output_embeddings(self, new_embeddings):
        self.classifier.emb2vocab = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = True if return_dict is None else return_dict

        sequence_output, contextualized = self.get_contextualized_embeddings(
            input_ids, attention_mask, output_hidden_states
        )
        logits = self.classifier(sequence_output)  # [B, T, vocab]

        loss = None
        if labels is not None:
            # Standard MLM: predict the token at each position (no shift). The
            # collator marks non-masked positions with -100 (cross_entropy's
            # default ignore_index), so loss is over the 15 % masked positions.
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + ((contextualized,) if output_hidden_states else ())
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized if output_hidden_states else None,
        )


def build_gpt_bert_rope_config(config, vocab_size: int | None = None) -> GptBertRopeConfig:
    """Map Prism's sizing dataclass (:class:`prism.bert.config.PrismBertConfig`)
    to a :class:`GptBertRopeConfig`, filling the architecture-specific fields
    with the vendored defaults (head size 64, GeGLU FF, RoPE, the 4:1
    local-global ratio).

    The attention windows are set to ``max_position_embeddings`` so the vendored
    local-global code path runs unchanged but is *full bidirectional attention*
    at our short pretraining/tagging sequence lengths — the long-context
    windowing the architecture supports, PrismBERT does not need.
    """
    from prism.bert.config import SPECIAL_TOKENS

    head_dim = config.hidden_size // config.num_attention_heads
    max_pos = config.max_position_embeddings
    return GptBertRopeConfig(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        query_key_head_size=head_dim,
        value_head_size=head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=vocab_size if vocab_size is not None else config.vocab_size,
        max_position_embeddings=max_pos,
        rope_theta=160_000,
        local_global_ratio=4,
        local_window_length=max_pos,
        global_window_length=max_pos,
        layer_norm_eps=1e-7,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        embedding_dropout=0.1,
        classifier_dropout=0.2,
        deterministic_flash_attn=False,
        use_cache=False,
        tie_word_embeddings=True,
        unk_token_id=SPECIAL_TOKENS["unk"][1],
        bos_token_id=SPECIAL_TOKENS["cls"][1],
        eos_token_id=SPECIAL_TOKENS["sep"][1],
        pad_token_id=SPECIAL_TOKENS["pad"][1],
        mask_token_id=SPECIAL_TOKENS["mask"][1],
    )


def register_gpt_bert_rope() -> None:
    """Idempotently register the arch with transformers' Auto classes, so a
    locally-saved backbone loads as its base encoder via
    ``AutoModel.from_pretrained`` (mirrors :func:`prism.bert.register`)."""
    global _registered
    if _registered:
        return
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    try:
        AutoConfig.register("gpt-bert-rope", GptBertRopeConfig)
    except ValueError:
        pass  # already registered in this interpreter
    try:
        AutoModel.register(GptBertRopeConfig, GptBertRopeEncoder)
    except ValueError:
        pass
    try:
        AutoModelForMaskedLM.register(GptBertRopeConfig, GptBertRopeForMaskedLM)
    except ValueError:
        pass
    _registered = True
