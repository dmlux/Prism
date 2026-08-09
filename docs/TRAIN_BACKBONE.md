# Training a PrismBERT backbone for your own language

PrismBERT is Prism's own quant-friendly encoder backbone: one architecture (the
LTG GPT-BERT design — DeBERTa-style relative attention, GeGLU, no RoPE, no
sliding window), **pretrained separately per language**, sized so the finished
tagger fits under **100 MB in fp32** and quantizes to a fast int8 `-fast`
variant on the ExecuTorch/XNNPACK CPU runtime. The design rationale and the
measurements behind it are in [PRISMBERT.md](PRISMBERT.md).

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
.venv/bin/python -m prism.prismbert.corpus --language en
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
`python/src/prism/prismbert/corpus.py` (pick openly-licensed, share-alike- or
public-domain text), then run with `--language <tag>`.

## 2. Train the tokenizer

```bash
.venv/bin/python -m prism.prismbert.tokenizer \
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
.venv/bin/python -m prism.prismbert.pretrain \
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
.venv/bin/python -u -m prism.prismbert.pretrain \
    --corpus data/pretraining/en --tokenizer models/prism-bert-en/tokenizer \
    --output runs/prism-bert-en --max-steps 100000 \
    2>&1 | tee -a "logs/prism-bert-en/pretrain-$(date +%Y%m%d-%H%M%S).log"
```

To let it run detached instead, wrap it in `nohup … &` and follow the log with
`tail -f logs/prism-bert-en/pretrain-*.log`. If it is interrupted, re-run the
same command with `--resume` to continue from the last checkpoint.

The default config is `PRISM_BERT_EN` in `config.py` — **H320 / 12 layers / 5
heads / FF 1024 / vocab 16384**, ~23.3 M backbone params, giving a ~99.6 MB fp32
tagger (backbone + heads + character CNN, measured). Masked-LM (15 %) via the HF
`Trainer` on MPS in fp32 (MPS has no reliable fp16/bf16).

Progress is logged as clean lines (tqdm disabled) every 100 steps
(`{'loss': …, 'learning_rate': …, 'epoch': …}`) and a held-out
`{'eval_loss': …}` every 2000 steps; checkpoints land in `runs/prism-bert-en/`
(`trainer_state.json` holds the full loss history for later plotting). Watch it
with `tail -f logs/prism-bert-en/pretrain-*.log`.

Throughput on the M4 Max: ~3.4 s/step (65 536 tokens/step) ≈ **19k tokens/s** →
**100 000 steps ≈ ~4 days (~6.5 B tokens ≈ ~1 epoch over the ~6 B corpus)** —
the recommended full run, since we stay under one epoch (no data repetition)
and the cosine LR schedule anneals over the whole budget (pick the target
upfront; a short run cannot be cleanly extended afterwards). Held-out
pseudo-perplexity is the intrinsic quality signal; the real gate is downstream
(step 5).

Hyperparameters are CLI flags: `--max-steps`, `--batch-size` (64), `--grad-accum`
(8), `--block-size` (128), `--learning-rate` (6e-4).

**For a new language:** add a `PrismBertConfig` for it in `config.py` sized to
the <100 MB budget (use the param-count approach in PRISMBERT.md), and point the
pretrain at that language's corpus + tokenizer.

## 5. Build the tagger on the backbone (downstream)

The pretrained backbone is an internal building block — it is **not shipped on
its own**; the released tagger (`prism-<lang>` fp32 + `-fast` int8) is distilled
onto it with Prism's existing pipeline, reusing a strong permissively-licensed
large model as the distillation teacher (English uses Ettin-encoder-400m, MIT —
no bespoke large model is trained). Package the backbone as a `transformers`
model, wire it as the student backbone, distil on gold + silver, then export
fp32 + int8 and gate against UDPipe 2.17. (This stage is being wired on the
`model/prism-bert-backbone` branch; this section will get exact commands as it
lands.)
