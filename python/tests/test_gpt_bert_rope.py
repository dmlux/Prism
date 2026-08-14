"""Tests for the RoPE GPT-BERT backbone (``prism.bert.gpt_bert_rope``) —
Prism's default PrismBERT architecture, vendored+adapted from LTG's GPT-BERT.

Fast, CPU-only, training-free: config mapping, weight tying + size budget,
standard (non-shifted) MLM loss semantics, and the save -> AutoModel-reload path
the tagger uses (with an explicit regression guard for the RoPE non-persistent
buffer NaN, see ``test_reloaded_encoder_is_finite``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prism.bert.config import PRISM_BERT_EN_ROPE, SPECIAL_TOKENS, PrismBertConfig
from prism.bert.gpt_bert_rope import (
    GptBertRopeConfig,
    GptBertRopeEncoder,
    GptBertRopeForMaskedLM,
    build_gpt_bert_rope_config,
    register_gpt_bert_rope,
)

# Tiny architecture for the fast forward/save/reload tests (head_size stays 64,
# which RoPE requires to be even).
_TINY = PrismBertConfig(
    hidden_size=64,
    num_layers=2,
    num_attention_heads=1,
    intermediate_size=128,
    vocab_size=64,
    max_position_embeddings=32,
)


def _tiny_masked_lm() -> GptBertRopeForMaskedLM:
    torch.manual_seed(0)
    return GptBertRopeForMaskedLM(build_gpt_bert_rope_config(_TINY))


def test_build_config_maps_dims_and_fills_arch_fields() -> None:
    cfg = build_gpt_bert_rope_config(PRISM_BERT_EN_ROPE, vocab_size=16384)

    assert cfg.model_type == "gpt-bert-rope"
    assert (cfg.hidden_size, cfg.num_layers, cfg.num_attention_heads) == (320, 14, 5)
    assert cfg.intermediate_size == 832
    assert cfg.vocab_size == 16384
    # head_size derived from hidden/heads; RoPE requires it even.
    assert cfg.query_key_head_size == cfg.value_head_size == 64
    # Windows are widened to full attention at our sequence lengths.
    assert cfg.local_window_length == cfg.global_window_length == cfg.max_position_embeddings
    assert cfg.tie_word_embeddings is True
    # Special-token ids match Prism's fixed convention.
    assert cfg.pad_token_id == SPECIAL_TOKENS["pad"][1]
    assert cfg.bos_token_id == SPECIAL_TOKENS["cls"][1]
    assert cfg.mask_token_id == SPECIAL_TOKENS["mask"][1]


def test_masked_lm_ties_embeddings_and_stays_within_budget() -> None:
    model = GptBertRopeForMaskedLM(build_gpt_bert_rope_config(PRISM_BERT_EN_ROPE, vocab_size=16384))

    # The MLM projection is the input embedding (shared Parameter object).
    assert model.classifier.emb2vocab.weight is model.embedding.word_embedding.weight

    backbone = sum(
        p.numel() for name, p in model.named_parameters() if not name.startswith("classifier.")
    )
    # The shipped English size: ~22.2 M backbone, the deepest config that keeps
    # the full fp32 tagger under the 100 MB budget. Pin it so size can't drift up
    # silently (backbone fp32 bytes must stay well under 100 MB on their own).
    assert 22.0e6 <= backbone <= 22.4e6
    assert backbone * 4 / 1e6 < 95.0


def test_masked_lm_loss_is_standard_not_shifted() -> None:
    model = _tiny_masked_lm().eval()
    input_ids = torch.randint(5, _TINY.vocab_size, (2, 8))

    # Supervise a single position; everything else is ignore_index.
    labels = torch.full((2, 8), -100)
    labels[0, 3] = 7

    out = model(input_ids=input_ids, labels=labels)

    assert out.logits.shape == (2, 8, _TINY.vocab_size)
    assert torch.isfinite(out.loss)
    # Standard (BERT) MLM predicts the token AT the masked position — not the
    # next one. So the loss must equal cross-entropy of logits at position 3,
    # NOT position 2 (which the vendored shifted causal head would use).
    expected = F.cross_entropy(out.logits[0, 3].unsqueeze(0), labels[0, 3].unsqueeze(0))
    torch.testing.assert_close(out.loss, expected)
    shifted = F.cross_entropy(out.logits[0, 2].unsqueeze(0), labels[0, 3].unsqueeze(0))
    assert not torch.allclose(out.loss, shifted)


def test_encoder_forward_is_finite_across_shapes() -> None:
    torch.manual_seed(0)
    encoder = GptBertRopeEncoder(build_gpt_bert_rope_config(_TINY)).eval()
    for batch, length in ((1, 1), (1, 7), (3, 16)):
        ids = torch.randint(5, _TINY.vocab_size, (batch, length))
        with torch.no_grad():
            hidden = encoder(input_ids=ids).last_hidden_state
        assert hidden.shape == (batch, length, _TINY.hidden_size)
        assert torch.isfinite(hidden).all()


def test_reloaded_encoder_is_finite(tmp_path) -> None:
    """Regression guard: RoPE cos/sin were non-persistent buffers computed in
    __init__, and transformers' meta-device ``from_pretrained`` left them
    uninitialised (NaN). They are now persistent; a fresh AutoModel reload — the
    path the tagger uses — must produce finite hidden states."""
    from transformers import AutoConfig, AutoModel

    model = _tiny_masked_lm()
    model.save_pretrained(tmp_path)

    # The deterministic RoPE tables must be persisted (the fix), not recomputed.
    from safetensors.torch import load_file

    saved = load_file(tmp_path / "model.safetensors")
    rope_keys = [k for k in saved if k.endswith(("cos_matrix", "sin_matrix"))]
    assert rope_keys, "RoPE buffers must be saved (persistent=True)"
    assert all(torch.isfinite(saved[k]).all() for k in rope_keys)

    register_gpt_bert_rope()
    register_gpt_bert_rope()  # idempotent — must not raise on the second call

    assert AutoConfig.from_pretrained(tmp_path).model_type == "gpt-bert-rope"
    encoder = AutoModel.from_pretrained(tmp_path).eval()
    assert isinstance(encoder, GptBertRopeEncoder)

    ids = torch.tensor([[SPECIAL_TOKENS["cls"][1], 9, 10, SPECIAL_TOKENS["sep"][1]]])
    with torch.no_grad():
        hidden = encoder(input_ids=ids).last_hidden_state
    assert torch.isfinite(hidden).all()


def test_config_subclass_has_stable_model_type() -> None:
    # The wrapper adds a stable model_type; the vendored base leaves it unset.
    assert GptBertRopeConfig.model_type == "gpt-bert-rope"
