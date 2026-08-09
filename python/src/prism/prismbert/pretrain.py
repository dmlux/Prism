"""Masked-language-model pretraining for PrismBERT on Apple Silicon (MPS).

Instantiates a FRESH (random) gpt_bert masked-LM at the chosen
:class:`~prism.prismbert.config.PrismBertConfig`, streams the JSONL corpus into
fixed-length token blocks, and trains with the standard 15 % MLM objective via
the HF ``Trainer`` on MPS (no CUDA on this machine). Held-out perplexity is the
intrinsic quality signal; the downstream UD gate (vs UDPipe) is the real bar.

There is no CUDA/distributed recipe here on purpose — a single-device MPS loop
is what this hardware supports. fp32 (MPS has only partial fp16/bf16 support);
a ~24 M model in fp32 fits the 64 GB unified memory comfortably.

Smoke (de-risk the loop before the multi-day run):
    PYTHONPATH=python/src .venv/bin/python -m prism.prismbert.pretrain \
        --corpus data/pretraining/en --tokenizer models/prism-bert-en/tokenizer \
        --output runs/prism-bert-en --smoke

Full run: drop --smoke and set --max-steps / --batch-size to taste.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from prism.prismbert.config import PRISM_BERT_EN, build_gpt_bert_config


class TokenBlockStream(IterableDataset):
    """Streams corpus shards, tokenizes, and yields fixed-length id blocks.

    Per-document chunking (no cross-document concatenation): each document's
    token ids are split into ``block_size`` blocks; the remainder is dropped.
    ``skip_shards`` / ``only_shards`` carve a held-out split by file.
    """

    def __init__(
        self,
        *,
        corpus_dir: Path,
        tokenizer,
        block_size: int,
        only_shards: list[Path] | None = None,
        skip_shards: set[str] | None = None,
        max_blocks: int | None = None,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.only_shards = only_shards
        self.skip_shards = skip_shards or set()
        self.max_blocks = max_blocks

    def _shards(self) -> list[Path]:
        if self.only_shards is not None:
            return self.only_shards
        return [
            s for s in sorted(self.corpus_dir.glob("*.jsonl"))
            if s.name not in self.skip_shards
        ]

    def __iter__(self) -> Iterator[dict]:
        emitted = 0
        for shard in self._shards():
            with shard.open(encoding="utf-8") as handle:
                for line in handle:
                    text = json.loads(line).get("text", "").strip()
                    if not text:
                        continue
                    ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                    for start in range(0, len(ids) - self.block_size + 1, self.block_size):
                        yield {"input_ids": ids[start : start + self.block_size]}
                        emitted += 1
                        if self.max_blocks is not None and emitted >= self.max_blocks:
                            return


def pretrain(args: argparse.Namespace) -> None:
    from transformers import (
        DataCollatorForLanguageModeling,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    from prism.prismbert.modeling_gpt_bert import GPTBERTForMaskedLM

    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available; falling back to CPU (very slow).")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    config = PRISM_BERT_EN  # the resolved default; edit config.py to change dims
    hf_config = build_gpt_bert_config(config)
    hf_config.vocab_size = tokenizer.vocab_size
    model = GPTBERTForMaskedLM(hf_config)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"Fresh PrismBERT: H{config.hidden_size}/L{config.num_layers}/"
        f"FF{config.intermediate_size}/V{tokenizer.vocab_size} — "
        f"{params/1e6:.1f}M params ({params*4/1e6:.0f} MB fp32 backbone).",
        flush=True,
    )

    all_shards = sorted(args.corpus.glob("*.jsonl"))
    if not all_shards:
        raise SystemExit(f"No .jsonl shards in {args.corpus}.")
    eval_shard = all_shards[-1]  # hold out the last shard for perplexity
    block = args.block_size
    train_ds = TokenBlockStream(
        corpus_dir=args.corpus, tokenizer=tokenizer, block_size=block,
        skip_shards={eval_shard.name},
        max_blocks=2000 if args.smoke else None,
    )
    eval_ds = TokenBlockStream(
        corpus_dir=args.corpus, tokenizer=tokenizer, block_size=block,
        only_shards=[eval_shard], max_blocks=200 if args.smoke else args.eval_blocks,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    max_steps = 50 if args.smoke else args.max_steps
    training_args = TrainingArguments(
        output_dir=str(args.output),
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, int(0.02 * max_steps)),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10 if args.smoke else 100,
        save_steps=args.save_steps,
        eval_steps=25 if args.smoke else args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        dataloader_num_workers=0,  # MPS + IterableDataset: keep single-process
        bf16=False,
        fp16=False,  # MPS AMP is unreliable; fp32 is safe and fits in 64 GB
        report_to=[],
        use_cpu=not torch.backends.mps.is_available(),
    )
    # The vendored GPTBERTForMaskedLM returns a correct HF masked-LM loss, so
    # the standard Trainer handles loss and gradient-accumulation scaling.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    print(f"Starting {'SMOKE ' if args.smoke else ''}pretraining on "
          f"{'MPS' if torch.backends.mps.is_available() else 'CPU'} "
          f"(block={block}, bs={args.batch_size}x{args.grad_accum})…", flush=True)
    trainer.train()
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    if loss is not None:
        import math
        print(f"held-out eval_loss={loss:.4f}  pseudo-perplexity={math.exp(loss):.2f}",
              flush=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved PrismBERT to {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--save-steps", type=int, default=2_000)
    parser.add_argument("--eval-steps", type=int, default=2_000)
    parser.add_argument("--eval-blocks", type=int, default=2_000)
    parser.add_argument("--smoke", action="store_true", help="Tiny run to de-risk the loop.")
    pretrain(parser.parse_args())


if __name__ == "__main__":
    main()
