# PrismBERT — a quant-friendly per-language backbone

PrismBERT is Prism's own encoder backbone: **one architecture**, pretrained
**separately per language** (`prismbert-en`, `prismbert-de`, …) as the Prism
tagger's backbone. Not multilingual — consistency lives in the architecture and
tooling, not in shared weights. Priority: on-device (ExecuTorch/XNNPACK CPU)
int8 **speed**, but quality must beat **UDPipe 2.17** per language. Size budget:
**< 100 MB fp32** per language (the int8 `-fast` variant is then ≈ ¼).

> **PrismBERT is the product name, not the architecture.** The architecture is
> LTG's *GPT-BERT with RoPE* — in this repo the `gpt_bert_rope` backbone.

UDPipe-2.17 English-EWT dev floor (gold-tokenised): **UPOS 97.56 / UFeats 97.86
/ Lemma 97.92**.

## Architecture: LTG GPT-BERT with RoPE (`gpt_bert_rope`)

The backbone is LTG's GPT-BERT design in its NorBERT4 generation: **RoPE**
positions, **local-global (sliding-window) attention**, **GeGLU** feed-forward,
a value-residual layer design, and parameter-free pre-LayerNorm. It is vendored
from `ltg/norbert4-base` (Apache-2.0) and adapted for Prism.

Code map (in `python/src/prism/bert/`):

| File | Origin | Role |
|------|--------|------|
| `modeling_gpt_bert_rope.py` | vendored (`ltg/norbert4-base`, modified) | the architecture |
| `configuration_gpt_bert_rope.py` | vendored (`ltg/norbert4-base`, modified) | the HF config |
| `gpt_bert_rope.py` | Prism-authored | budget-sized config, standard-MLM head, encoder, `AutoModel` registration |

The Prism adapter exposes `GptBertRopeConfig` (model_type `gpt-bert-rope`),
`GptBertRopeEncoder` (what the tagger loads via `AutoModel`),
`GptBertRopeForMaskedLM` (the pretraining head), `build_gpt_bert_rope_config`,
and `register_gpt_bert_rope`. It is covered by `python/tests/test_gpt_bert_rope.py`.

NorBERT4 is only where this architecture is *published*; the vendored file
headers carry the attribution. The Norwegian tagger separately uses the actual
`ltg/norbert4-*` checkpoints — that is a different, orthogonal choice.

## Why this architecture

Measured on the shipped models with identical export/int8 tooling:

- **ModernBERT/Ettin (English)** is hostile to low precision on the XNNPACK CPU
  path: int8 ends up **slower** than fp32 (≈ +6 % even with the grouped
  partitioner) and −1.7 pp; fp16 collapses −17 pp.
- **NorBERT4 (Norwegian, LTG GPT-BERT family)** with the SAME tooling is int8
  **≈ 1.9× faster** than fp32 and near-lossless (`prism-no-…-fast`), and with our
  Norwegian data it **beat UDPipe 2.17 in 4 of 6** metrics.

The earlier working hypothesis — "avoid RoPE and sliding-window, they are
int8-hostile" — was **wrong** and is corrected here: NorBERT4 *has* RoPE and
local-global attention and is still int8-fast and near-lossless. The real lever
is a **clean implementation** (no in-place custom-autograd tricks, no
export-hostile ops), not the positional scheme. Ettin's int8 problems were its
specific implementation, not RoPE per se.

So we rebuilt PrismBERT on NorBERT4's architecture and pretrain **our own
weights per language on our own clean corpus** — only the architecture template
changed; the per-language, quant-friendly thesis is unchanged. The deciding
reason, confirmed by the English A/B below, is **int8 deployability**: this arch
is the one that produces a working, fast int8 `.pte`. (On English it does *not*
win on fp32 quality — see the A/B — but int8 deployment is the hard constraint,
and there it is the only arch that works end-to-end.)

## Training objective: standard bidirectional MLM

PrismBERT is pretrained with **standard bidirectional masked-LM** (15 % masking),
not GPT-BERT's mixed shifted causal/MLM objective. Reason: the tagger consumes
the backbone **bidirectionally** (each token attends left and right), which is
exactly what MLM trains; the causal half of the GPT-BERT recipe mainly helps
*generation*, which we do not use. Our first PrismBERT (on the legacy arch) was
already MLM-trained and reached ~parity with the silver-trained Ettin student,
so we keep the proven recipe and change only the architecture.

The vendored `GptBertForMaskedLM` (the shifted causal/MLM head, with
`30·sigmoid` logit-bounding and a BOS-prepend) is kept untouched for reference;
Prism uses `GptBertRopeForMaskedLM` instead — a plain, non-shifted MLM head with
tied input/output embeddings.

## Sizing — the < 100 MB budget

`PRISM_BERT_EN_ROPE` (in `prism/bert/config.py`): **hidden 320 / 14 layers / 5
heads / head-size 64 / FF 832 / vocab 16384** → **22.2 M backbone (88.8 MB
fp32)**, and a **~95 MB fp32** full tagger (backbone + task heads + character
CNN). This is the deepest configuration that stays under the 100 MB budget: same
width/heads as the legacy config, two layers deeper at the arch's ~2.6× GeGLU FF
ratio. The size is pinned by a test so it cannot drift up silently.

For a new language, add a `PrismBertConfig` sized to the budget and point the
pretraining at that language's corpus + tokenizer.

## Pretraining

Everything lives in the `prism/bert/` package; the reproducible recipe and exact
commands are in [TRAIN_BACKBONE.md](TRAIN_BACKBONE.md). In brief:

- **Corpus** (`prism.bert.corpus`): legally-clean, commercial-safe — English
  Wikipedia (CC BY-SA 3.0/GFDL) + Project Gutenberg (public domain). No
  CommonCrawl-derived text. The reader **interleaves the sources** (deterministic,
  resume-safe) so every register is present throughout training and the LR
  anneal.
- **Tokenizer** (`prism.bert.tokenizer`): byte-level BPE, vocab 16384, special
  tokens fixed at `<unk>`=0/`<s>`=1/`</s>`=2/`<pad>`=3/`<mask>`=4.
- **Pretraining** (`prism.bert.pretrain --arch gpt_bert_rope`, the default):
  fresh masked-LM via the HF `Trainer`. On this machine (Apple M4 Max, MPS, no
  CUDA) training is fp32 at **~2.75 s/step** (65 536 tokens/step) → **100 000
  steps ≈ 3.5–4 days ≈ 6.6 B tokens ≈ ~1 epoch**. Resume-safe (`--resume`);
  best-eval checkpoint kept + reloaded; progress logged via the shared
  `prism.progress` logger. A `--smoke` run de-risks the loop first.
- **Teacher (no own large model):** reuse **Ettin-encoder-400m (MIT)** as the
  distillation teacher — a permissive large model keeps the student commercially
  safe. Same pattern per language: pick a strong permissive teacher, don't
  pretrain one.

The legacy BabyLM GPT-BERT arch (DeBERTa-style disentangled relative positions,
no RoPE) is still available as `--arch gpt_bert` (config `PRISM_BERT_EN`), kept
for comparison; it is no longer the default.

## Implementation notes / gotchas

- **RoPE buffers are persistent.** Upstream registers the RoPE cos/sin tables as
  `persistent=False` and recomputes them per load. transformers 5.x
  `from_pretrained` builds on the *meta* device and does **not** recompute
  `__init__`-time non-persistent buffers, leaving them as uninitialised memory
  (NaN) on any fresh load (the tagger, or any `AutoModel.from_pretrained`).
  PrismBERT makes them **`persistent=True`** (a marked vendored modification; a
  regression test guards it). Pretraining/`--resume` were never affected (the
  model is built materialised there).
- **FlashAttention is optional.** The vendored modeling uses FlashAttention only
  if available (CUDA); otherwise it runs a portable eager path (the one we use on
  MPS/CPU and for ExecuTorch export). No `flash-attn` dependency.
- **Windows widened to full attention.** `local_window_length` and
  `global_window_length` are set to `max_position_embeddings`, so the vendored
  local-global code path runs unchanged but is full bidirectional attention at
  our short tagging/pretraining sequence lengths — the long-context windowing is
  not needed here.

## The deploy decision: RoPE vs the legacy BabyLM arch (2026-08-20)

Both PrismBERT archs were distilled through the identical English pipeline
(teacher Ettin-400m, gold + KD, no silver, 12 epochs, dev-loss selection) and
measured head-to-head. The RoPE arch (`gpt_bert_rope`) is **locked as the deploy
architecture** — it is the only one that yields a working, fast int8 model.

| Axis | BabyLM (`gpt_bert`, 23.3 M) | RoPE (`gpt_bert_rope`, 22.2 M) | Winner |
|------|-----------------------------|-------------------------------|--------|
| fp32 dev UD-F1 (UPOS/UFeats/Lemma) | 97.08 / 97.01 / 97.52 | 96.68 / 96.63 / 97.37 | BabyLM (+~0.4 pp) |
| int8 eager quality (worst Δ vs fp32) | −0.04 pp | −0.32 pp (morphology) | BabyLM |
| int8 `.pte` size | 26.7 MiB | 24.3 MiB | ~tie |
| **int8 `.pte` builds** | yes (after the fix below) | yes | — |
| **int8 runs on the native runtime** | **no — silent abort** | **yes** | **RoPE** |
| int8 speed (native, 1 thread, 6 k-tok doc) | n/a (crashes) | **412 tok/s = 2.0× its fp32** | **RoPE** |

**Why BabyLM int8 is a dead end here.** Its DeBERTa-style disentangled relative
attention applies a linear to the constant `relative_embedding`; the XNNPACK
dynamic-quant partitioner does not claim it, leaving an un-delegated per-channel
quantize in the graph. That op now *lowers* (see the export fix below) but
*aborts at native runtime* — the same int8-hostile pattern the RoPE arch avoids
by construction (no disentangled attention → the embedding fuses to
`embedding_byte` and every linear delegates). Neither arch beats UDPipe on
gold+KD-only yet; closing that gap is the silver-data step, which helps both and
is orthogonal to this decision.

## int8 + release

**Export blocker fixed (2026-08-20).** `to_executorch` previously failed on any
un-delegated per-channel quantize with "Missing out variants:
`quantized_decomposed::quantize_per_channel`". ExecuTorch ships portable `.out`
kernels for `{quantize,dequantize}_per_channel` (its C++ runtime) but does not
register them as Python torch ops, so the AOT `to_out_var` pass could not resolve
them. `lower_to_executorch_xnnpack` now registers those `.out` overloads before
lowering (idempotent; a no-op for the fully-delegated RoPE graph). This unblocks
the *export*; note that an un-delegated per-channel quantize still needs a
runtime kernel that executes cleanly — it does for RoPE (nothing un-delegated),
but the BabyLM graph aborts (above).

Near-lossless post-training int8 holds for RoPE (worst −0.32 pp, morphology);
QAT (`torchao prepare_qat_pt2e` → ExecuTorch XNNPACK) stays a fallback only if
the int8 gate regresses.

Ship the tagger fp32 + int8 (`-fast`) **from `main`** after the UD gate vs
UDPipe 2.17; publish the raw PrismBERT backbone separately on HF (not bundled in
the tagger tarball). Weights release under **CC BY-SA 4.0** (the SA chain is
honoured by CC-BY-SA Wikipedia + public-domain Gutenberg); code under Apache-2.0.

## Next step: silver-data distillation

Gold+KD alone leaves both archs below UDPipe. The next quality lever is
silver-data distillation on the locked RoPE backbone: teacher-label a large
clean English corpus with Ettin-400m and add it to the student's training.
Tooling: `prism.languages.english.prepare_silver_corpus` +
`label_silver_corpus`, then `train_baseline --student-backbone prismbert-en
--silver-corpus … --silver-labels …`. This is how the shipped Ettin student got
its gains; it applies unchanged to the RoPE backbone.

## History

- **Stage-1 (2026-08-09):** validated the GPT-BERT integration + int8 delegation
  on a BabyLM checkpoint through the Prism pipeline (linears fully delegate to
  XNNPACK) — arch + plumbing proven before investing in pretraining.
- **Legacy PrismBERT-en** (BabyLM gpt-bert, `PRISM_BERT_EN` = H320/L12/FF1024,
  23.3 M): MLM-pretrained, distilled → UD dev UPOS 97.08, ~parity with the
  silver Ettin-17m student. Its int8 collapse (−53 pp) traced to the DenseFormer
  DWA's in-place custom-autograd and was fixed by a functional rewrite
  (fp32 bit-identical, int8 near-lossless).
- **2026-08-14:** rebuilt PrismBERT on NorBERT4's architecture (`gpt_bert_rope`),
  made it the default, and pretrained `prismbert-en` (H320/L14/FF832, 22.2 M,
  held-out ppl 5.12).
- **2026-08-20:** distilled RoPE and legacy-BabyLM students head-to-head and
  measured int8 export/size/speed/runtime. Fixed the int8 `.pte` export blocker
  (portable per-channel-quant `.out` variants). Found BabyLM int8 aborts at
  native runtime while RoPE int8 runs at 2.0× its fp32; **locked RoPE as the
  deploy architecture** despite its ~0.4 pp fp32 deficit. Next: silver-data
  distillation (both archs still trail UDPipe on gold+KD only).
