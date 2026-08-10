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

### Stage-1 smoke-test result (2026-08-09) — plumbing + int8 delegation ✅

Built a fresh, untrained Prism tagger (LINEAR heads, random) on the BabyLM
backbone and ran it through the production int8 lowering
(`scratchpad/babylm_int8_probe.py`):

- **Plumbing works end-to-end:** `build_pretrained_token_tagger(backbone_spec=
  babylm, schema=en-2.17, …)` builds (83 `nn.Linear`), the backbone tokenizer
  loads, and the eager forward runs — with only the two documented handles
  (`reinitialize_non_persistent_buffers=True` + the `all_tied_weights_keys`
  shim). Fold is a no-op (0 scale-parametrized linears).
- **int8 delegation:** in the lowered edge graph, `aten_linear_default` is
  **95 delegated / 0 non-delegated** — every linear goes to XNNPACK, no
  portable compute fallback. Same int8-friendly profile as NorBERT4 (whose
  int8 we measured at 1.9× fp32). 149 grouped subgraphs.
- **Open follow-up (not a delegation issue):** `to_executorch` currently fails
  with `Missing out variants: quantized_decomposed::quantize_per_channel` — a
  non-delegated per-channel quantize (likely on the embedding path) whose
  portable out-variant is unregistered in this ET build. NorBERT4 completes
  `to_executorch` (it fuses this into `embedding_byte`), so it is a solvable
  export-config/pass detail to resolve when wiring the real PrismBERT export,
  not a blocker. Reference: the NorBERT4 int8 export path.

**Conclusion:** the GPT-BERT-en architecture integrates into the Prism pipeline
and its int8 linears fully delegate to XNNPACK — Weg A validated. Proceed to
Stage 2 (pretrain), resolving the `to_executorch` out-variant during real
export wiring.

## Stage 2 — pretrain the deployable PrismBERT-en (in `prism.bert`)

Stage 1 held, so we pretrain our own backbone. Everything lives in the
`prism/bert/` package.

**Config (finalized, measured):** `PRISM_BERT_EN` = hidden 320, 12 layers, 5
heads, FF 1024, vocab 16384. The FULL tagger (backbone + heads + character CNN +
structured morphology + lemma head) is **~99.6 MB fp32** — under the 100 MB
budget and well above today's Ettin-17m (68.6 MB), using the headroom for
capacity. Deeper/narrower chosen over wider/shallower (H384/8L, 95.7 MB) for
morphology/syntax and alignment with the deep NorBERT4 family the Prism heads
are tuned on; the wide variant is the fallback if UFeats disappoints.

**Corpus (legally clean, commercial-safe):** `prism.bert.corpus` streams
English Wikipedia (CC BY-SA 3.0/GFDL, ~3B tokens, modern register) + Project
Gutenberg (`sedthh/gutenberg_english`, public domain, literary register — the
primary Prism use case) to JSONL shards with pinned dataset revisions. No
CommonCrawl-derived text (unclear copyright). ~4–5B clean tokens — ample for a
~24M model. Weights release under CC BY-SA 4.0; the SA chain is honored by the
CC-BY-SA Wikipedia + public-domain Gutenberg provenance.

**Tokenizer:** `prism.bert.tokenizer` — byte-level BPE, vocab 16384,
special tokens fixed to the gpt_bert ids.

**Pretraining:** `prism.bert.pretrain` — fresh gpt_bert masked-LM, 15% MLM
via HF Trainer. **This machine is Apple M4 Max (40-core GPU, 64 GB), MPS only —
no CUDA**, so training is fp32 on MPS, days-to-weeks (not the 48-GPU-h CUDA
reference), run iteratively (start ~1–2B tokens → measure → extend). Track
held-out pseudo-perplexity; a `--smoke` run de-risks the loop first.

**Teacher (no own large model):** reuse the existing **Ettin-encoder-400m (MIT)**
as the distillation teacher — permissive license keeps the student commercially
safe; cross-architecture logit + silver distillation is fine. Same pattern for
other languages: pick a strong permissive large model, don't pretrain a teacher.

**int8 + ship:** near-lossless post-training int8 expected (NorBERT4 family);
QAT (`torchao prepare_qat_pt2e` → ExecuTorch XNNPACK) only as a fallback if the
int8 gate regresses. Resolve the Stage-1 `to_executorch` out-variant during
export wiring. Distil the tagger (existing pipeline, student backbone swapped)
on gold + silver, then release fp32 + int8(`-fast`) **from `main`**; publish the
raw PrismBERT-en backbone separately on HF (not bundled in the tagger tarball).
