# Training a PrismBERT backbone for your own language

PrismBERT is Prism's own quant-friendly encoder backbone: one architecture
(LTG's GPT-BERT design with **RoPE**, local-global (sliding-window) attention and
GeGLU — the `gpt_bert_rope` backbone, vendored from `ltg/norbert4-base`),
**pretrained separately per language**, sized so the finished tagger fits under
**100 MB in fp32** and quantizes to a fast int8 `-fast` variant on the
ExecuTorch/XNNPACK CPU runtime. The design rationale and the measurements behind
it are in [PRISMBERT.md](PRISMBERT.md).

This guide is the reproducible, end-to-end recipe. Every command below was run
to produce the English backbone; the numbers are from an Apple M4 Max (40-core
GPU / 64 GB, MPS — no CUDA on this machine).

> Model weights trained with this recipe should be released under **CC BY-SA
> 4.0** (Prism's convention). Keep the corpus legally clean (see step 1) so the
> release is commercial-use-safe.

## 0. Prerequisites

```bash
# from the repository root, in the project venv
.venv/bin/pip install datasets accelerate            # tokenizers ships with transformers
export PYTHONPATH=python/src
```

All artifacts go to gitignored directories: the corpus to `data/pretraining/`,
model outputs to `runs/`, and training logs to `logs/`.

## 1. Download a legally-clean corpus

```bash
.venv/bin/python -m prism.bert.corpus --language en
```

Streams openly-licensed sources to `data/pretraining/en/*.jsonl` and writes a
`provenance.json` pinning each dataset's exact revision + license. For English:
Wikipedia (`wikimedia/wikipedia` 20231101.en, CC BY-SA 3.0/GFDL — modern
register) and Project Gutenberg (`sedthh/gutenberg_english`, public domain —
literary register). Result: ~5.9B whitespace-tokens, ~37 GB. It is **idempotent**
(re-running skips sources whose shards already exist) and per-source resilient.

CommonCrawl-derived corpora (FineWeb / C4 / OSCAR) are deliberately excluded —
their copyright status is unclear and incompatible with a commercial guarantee.

**For a new language:** register its sources in `SOURCES_BY_LANGUAGE` in
`python/src/prism/bert/corpus.py` (pick openly-licensed, share-alike- or
public-domain text), then run with `--language <tag>`.

## 2. Train the tokenizer

```bash
.venv/bin/python -m prism.bert.tokenizer \
    --corpus data/pretraining/en --output models/prism-bert-en/tokenizer \
    --vocab-size 16384 --sample-tokens 500000000
```

Trains a byte-level BPE (robust to modern/OOV text) with the five special
tokens fixed at the ids the model expects (`<unk>`=0, `<s>`=1, `</s>`=2,
`<pad>`=3, `<mask>`=4). Sources are interleaved and capped by a token budget so
token-dense books don't dominate the vocabulary. Saves a `transformers`
fast-tokenizer directory (`tokenizer.json` + configs). Takes a few minutes.

## 3. Smoke-test the training loop (recommended)

```bash
.venv/bin/python -m prism.bert.pretrain --arch gpt_bert_rope \
    --corpus data/pretraining/en --tokenizer models/prism-bert-en/tokenizer \
    --output runs/prism-bert-en-smoke --smoke
```

50 steps + a held-out eval, to confirm the loop runs on your hardware, the loss
is sane (~9.7 = ln(vocab) at init), and the model saves + reloads. De-risks the
multi-day run.

## 4. Pretrain

```bash
mkdir -p logs/prism-bert-en
# `-u` unbuffered + `tee -a` -> progress prints live to the terminal AND is
# appended live to a persistent (gitignored) log file at the same time.
.venv/bin/python -u -m prism.bert.pretrain --arch gpt_bert_rope \
    --corpus data/pretraining/en --tokenizer models/prism-bert-en/tokenizer \
    --output runs/prism-bert-en --max-steps 100000 \
    2>&1 | tee -a "logs/prism-bert-en/pretrain-$(date +%Y%m%d-%H%M%S).log"
```

To let it run detached instead, wrap it in `nohup … &` and follow the log with
`tail -f logs/prism-bert-en/pretrain-*.log`. If it is interrupted, re-run the
same command with `--resume` to continue from the last checkpoint.

The default architecture is `gpt_bert_rope` (LTG's GPT-BERT with RoPE); its
config is `PRISM_BERT_EN_ROPE` in `config.py` — **H320 / 14 layers / 5 heads /
FF 832 / vocab 16384**, ~22.2 M backbone params, giving a ~95 MB fp32 tagger
(backbone + heads + character CNN, measured). Standard bidirectional masked-LM
(15 %) via the HF `Trainer` on MPS in fp32 (MPS has no reliable fp16/bf16). Pass
`--arch gpt_bert` to train the legacy BabyLM GPT-BERT instead (DeBERTa-style
relative positions, config `PRISM_BERT_EN`).

Progress is logged as aligned, table-like lines (tqdm disabled): labels spelled
out, numbers blank-padded with fixed decimals, and the tokens consumed so far
shown next to the step so you never have to back-compute it from the percentage.
A `[train]` row every 100 steps and a held-out `[eval ]` row (with throughput)
every 2000 steps:

```text
[train] step   2000/100000    tokens   131,072,000    loss           2.587000    gradient_norm      0.608800    learning_rate      0.000600    epoch                   0.020000
[eval ] step   2000/100000    tokens   131,072,000    eval_loss      5.744000    perplexity       312.311160    eval_seconds       8.344000    samples_per_second    239.700000    steps_per_second     29.960000
```

Checkpoints land in `runs/prism-bert-en/` (`trainer_state.json` holds the full
loss history for later plotting); the **best-eval checkpoint is kept** (protected
from `save_total_limit` pruning) and **reloaded at the end**, so the saved model
is the best one, not merely the last. Watch it with
`tail -f logs/prism-bert-en/pretrain-*.log`.

Throughput on the M4 Max: ~2.75 s/step (65 536 tokens/step) ≈ **24k tokens/s** →
**100 000 steps ≈ ~3.5–4 days (~6.6 B tokens ≈ ~1 epoch over the ~6 B corpus)** —
the recommended full run, since we stay under one epoch (no data repetition)
and the cosine LR schedule anneals over the whole budget (pick the target
upfront; a short run cannot be cleanly extended afterwards). Held-out
pseudo-perplexity is the intrinsic quality signal; the real gate is downstream
(step 5). The reader **interleaves the corpus sources** (weighted by size,
deterministically — so `--resume` stays reproducible) instead of reading one
source fully and then the next, keeping every register present throughout
training and the LR anneal; the held-out eval reserves the first `--eval-blocks`
blocks of **each** source, so the perplexity reflects all registers (not just
whichever shard sorts last).

The architecture is selected with `--arch {gpt_bert_rope (default), gpt_bert}`
(`gpt_bert` = the legacy BabyLM DeBERTa-relative backbone). Hyperparameters are
CLI flags: `--max-steps`, `--batch-size` (64), `--grad-accum` (8), `--block-size`
(128), `--learning-rate` (6e-4). Two knobs exist for portability to non-Apple
hardware: `--precision {fp32,bf16,fp16}` (default fp32;
bf16/fp16 speed up CUDA a lot but are unreliable on MPS) and `--num-workers`
(default 0 = inline tokenization, best on MPS where training is GPU-bound; `>0`
shards the corpus across dataloader workers, for when CPU tokenization is the
bottleneck).

**For a new language:** add a `PrismBertConfig` for it in `config.py` sized to
the <100 MB budget (use the param-count approach in PRISMBERT.md), and point the
pretrain at that language's corpus + tokenizer.

## 5. Build the tagger on the backbone (downstream)

The pretrained backbone is an internal building block — it is **not shipped on
its own**; the released tagger (`prism-<lang>` fp32 + `-fast` int8) is distilled
onto it with Prism's existing pipeline, reusing a strong permissively-licensed
large model as the distillation teacher (English uses Ettin-encoder-400m, MIT —
no bespoke large model is trained).

The English backbone is wired as a selectable student. Distil (gold + KD), then
evaluate the dev UD-F1 vs the UDPipe-2.17 floor:

```bash
# distil the tagger on the pretrained runs/prismbert-en backbone
PYTHONPATH=python/src caffeinate -is .venv/bin/python -u -m prism.languages.english.train_baseline \
    --model-role student --student-backbone prismbert-en --treebank-release 2.17 \
    --teacher-checkpoint runs/en-teacher-400m/best.pt \
    --token-pooling mean \
    --task-head-architecture wide-shared-mlp-structured-morphology-character-cnn \
    --morphology-pre-head-architecture shared-mlp \
    --backbone-layer-aggregation learned-last-four \
    --epoch-count 12 --early-stopping-patience 4 \
    --checkpoint-selection-metric development-loss \
    --checkpoint runs/en-prismbert-rope-student/best.pt

# dev UD-F1 (add --silver-* once silver data is prepared, see below)
PYTHONPATH=python/src .venv/bin/python -m prism.languages.english.evaluate_baseline \
    --treebank-release 2.17 --split development \
    --checkpoint runs/en-prismbert-rope-student/best.pt \
    --analysis runs/en-prismbert-rope-student/development-analysis.json \
    --morphology-logit-correction-strength 0.0 --device mps
```

Export fp32 + int8 (`.pte`) and measure size/speed (see
[PRISMBERT.md](PRISMBERT.md) for the A/B results and the deploy decision):

```bash
PYTHONPATH=python/src .venv/bin/python -m prism.languages.english.export_artifact \
    --checkpoint runs/en-prismbert-rope-student/best.pt --output-root models \
    --artifact-version rope-int8 --treebank-release 2.17 --precision int8 \
    --morphology-logit-correction-strength 0.0
```

**Silver-data distillation** is the quality lever that pushes the RoPE tagger
past BabyLM and close to UDPipe (dev UD-F1 97.34 / 97.52 / 97.65; see
[PRISMBERT.md](PRISMBERT.md)). The English silver corpora are already
teacher-labelled by Ettin-400m (5 M tokens each), so no re-labelling is needed —
just add them to the distillation:

```bash
PYTHONPATH=python/src caffeinate -is .venv/bin/python -u -m prism.languages.english.train_baseline \
    --model-role student --student-backbone prismbert-en --treebank-release 2.17 \
    --teacher-checkpoint runs/en-teacher-400m/best.pt \
    --token-pooling mean \
    --task-head-architecture wide-shared-mlp-structured-morphology-character-cnn \
    --morphology-pre-head-architecture shared-mlp \
    --backbone-layer-aggregation learned-last-four \
    --epoch-count 12 --early-stopping-patience 4 \
    --checkpoint-selection-metric development-task-accuracy \
    --secondary-checkpoint-selection-metric development-loss \
    --silver-corpus data/processed/gutenberg-eng/pretokenized.jsonl \
    --silver-labels data/processed/gutenberg-eng/labels \
    --silver-corpus data/processed/wikipedia-eng/pretokenized.jsonl \
    --silver-labels data/processed/wikipedia-eng/labels \
    --silver-loss-weight 0.5 --silver-disable-agreement-filter \
    --checkpoint runs/en-prismbert-rope-silver/best.pt
```

Three silver-specific flags matter:

* **`--checkpoint-selection-metric development-task-accuracy`** (with
  `--secondary-checkpoint-selection-metric development-loss` to keep both). Under
  silver, dev-loss and dev-accuracy *diverge* — loss bottoms early (~epoch 2)
  and rises while accuracy keeps climbing (~epoch 11). Selecting by loss ships a
  markedly worse checkpoint (97.02 / 97.08 / 97.34 vs 97.34 / 97.52 / 97.65).
* **`--silver-disable-agreement-filter`** — these labels carry no second-model
  agreement predictions, so the default agreement filter would error. (Re-label
  with `label_silver_corpus --agreement-checkpoint …` if you want that filter.)
* Silver sentences longer than the backbone's `max_position_embeddings` (512
  subwords) are dropped automatically — the RoPE tables are that long, and long
  Gutenberg/Wikipedia sentences would otherwise overflow them.

To prepare silver for a **new** language: `prepare_silver_corpus` (extract
sentences from a raw archive) then `label_silver_corpus` (label them with the
teacher + its calibration).

The architecture decision is settled: the released English tagger uses the
**RoPE backbone (`--student-backbone prismbert-en`)** with silver — the only arch
with a working, fast int8 `.pte` (see PRISMBERT.md). The legacy BabyLM backbone
(`--student-backbone prism-bert-en`) is kept for comparison only.
