# PrismBERT — a quant-friendly per-language backbone

Working document for the `model/prism-bert-backbone` branch. Goal: fix ONE
modern, int8-quantization-friendly encoder architecture and pretrain it
**separately per language** (PrismBERT-en, PrismBERT-no, …) as the Prism
tagger backbone. Not multilingual — consistency lives in the architecture,
not shared weights. Priority: on-device (ExecuTorch/XNNPACK CPU) int8 **speed**
over quality, but quality must beat **UDPipe 2.17** per language. Size budget:
**< 100 MB fp32** per language (int8 `-fast` then ≈ ¼).

## Why a new backbone

Measured on the shipped models with the identical export/int8 tooling:

- **ModernBERT/Ettin (English)** is hostile to low precision on the XNNPACK CPU
  path: int8 ends up **slower** than fp32 (≈ +6 % even with the grouped
  partitioner) and −1.7 pp; fp16 collapses −17 pp. Cause: RoPE +
  sliding-window / alternating global-local attention.
- **NorBERT4 (Norwegian, LTG/GPT-BERT family)** with the SAME tooling is int8
  **≈ 1.9× faster** than fp32 and near-lossless (prism-no-0.2.5-fast).

So the backbone architecture is the lever. **Blueprint: LTG GPT-BERT**
(Charpentier & Samuel 2024, BabyLM winner; same group as NorBERT4). Quant-
friendly by construction: DeBERTa-style disentangled **relative** positions
(`position_bucket_size` bucketing — NOT RoPE, NOT learned-absolute), full
bidirectional attention (no sliding-window / global-local), GeGLU,
parameter-free pre-LayerNorm. Repo: <https://github.com/ltgoslo/gpt-bert>,
paper arXiv:2410.24159.

## Stage 1 — validate the architecture without pretraining (Weg A)

Distill/train an existing English GPT-BERT through the current Prism pipeline,
export int8, and check (a) the int8 speed win and (b) whether it clears the
UDPipe-2.17-en floor. Validation only — the base checkpoint is over the size
budget; it proves the arch + integration + quality-reachability before we
invest in pretraining.

**Chosen vehicle: `BabyLM-community/babylm-baseline-100m-gpt-bert-mixed`**
(rev `09629ffe557c4143aa7b857f92004f3e45689eff`, 118.8 M, hidden 768 / 12 layers
/ vocab 16384 / `position_bucket_size` 32). It ships the **canonical
`modeling_gpt_bert.py`** — the exact code path we will reuse when we pretrain
our own PrismBERT in Stage 2 — so validating it also validates that path.

Smoke test (2026-08-09) confirmed integration in this repo's env
(transformers 5.13.1):

- Loads via `AutoModel.from_pretrained(..., trust_remote_code=True)` +
  **`reinitialize_non_persistent_buffers=True`** (same as NorBERT4 — the
  non-persistent `position_indices` buffer must be rebuilt, else the forward
  raises `IndexError` in the relative-position embedding lookup).
- Needs a one-line transformers-5.x shim: `PreTrainedModel.all_tied_weights_keys
  = {}` (the canonical code predates that attribute). NorBERT4's newer
  `modeling_gptbert.py` does not need it.
- Forward returns `last_hidden_state` + 13 `hidden_states` → tagger-compatible.
- Modules are plain `nn.Linear` / `GeGLU` / `Attention` — **no scale-
  parametrized linears**, so the int8 fold is a no-op and the standard
  `xnnpack-embedding-dynamic` strategy applies directly (no ModernBERT mask
  surgery).

**Avoid** `ltg/gpt-bert-babylm-base` and `ltg/gpt-bert-babylm-small`: both ship
the older `modeling_ltgbert.py`, whose forward is incompatible with
transformers 5.13 (`config.is_decoder` etc.). The small one (hidden 384, ~30 M)
is budget-sized but we will pretrain our own with canonical code anyway.

### Stage-1 steps

1. Wire a `PretrainedBackboneSpec` for the vehicle (trust_remote_code,
   reinitialize_non_persistent_buffers) + the `all_tied_weights_keys` shim in
   the backbone loader; add an English GPT-BERT profile using
   `quantization="xnnpack-embedding-dynamic"`.
2. int8-delegation probe on a (randomly-initialised-head) adapter → confirm the
   linears fully delegate to XNNPACK (expected, like NorBERT4).
3. Distill/train through the existing English pipeline (**user runs** this —
   expensive) → student checkpoint.
4. Export int8 + fp32 → measure C++ speed (expect int8 > fp32) and UD dev
   quality vs the UDPipe-2.17-en floor (**UPOS 97.56 / UFeats 97.86 /
   Lemma 97.92**).

## Stage 2 — pretrain the deployable PrismBERT-en (~30 M)

If Stage 1 holds: pretrain a strict-small-class config (hidden 384, 12 layers,
FF 1280, 6 heads, small vocab) with the ltgoslo/gpt-bert recipe + **QAT**
(torchao `prepare_qat_pt2e` → ExecuTorch XNNPACK), on English raw text. ~48
GPU-hours for the 120 M baseline → a smaller model + one strong GPU is a few
days. Confirm tied embeddings to stay < 100 MB fp32. Then distil the Prism
tagger onto it and release fp32 + int8(`-fast`) from `main`.
