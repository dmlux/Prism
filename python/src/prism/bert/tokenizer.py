"""Train the PrismBERT byte-level BPE tokenizer on the pretraining corpus.

Byte-level BPE is robust to any input (no real UNK, covers modern/OOV text —
important given the corpus mixes historical books with modern Wikipedia). The
five special tokens are placed at the fixed ids the gpt_bert architecture and
the MLM collator expect (``<unk>``=0, ``<s>``=1, ``</s>``=2, ``<pad>``=3,
``<mask>``=4). The result is saved as a ``transformers`` fast-tokenizer
directory (``tokenizer.json`` + configs) that both the pretraining loop and the
downstream Prism pipeline (``load_backbone_tokenizer``) can load.

A BPE vocabulary does not need the full corpus; training on a representative
sample is standard and much faster. ``--sample-tokens`` caps it and the sources
are interleaved so both registers (modern Wikipedia + literary Gutenberg) are
represented rather than token-dense books dominating.

Run:
    PYTHONPATH=python/src .venv/bin/python -m prism.bert.tokenizer \
        --corpus data/pretraining/en --output models/prism-bert-en/tokenizer \
        --vocab-size 16384 --sample-tokens 500000000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from prism.bert.config import SPECIAL_TOKENS

_SPECIAL_ORDER = ["unk", "cls", "sep", "pad", "mask"]  # -> ids 0,1,2,3,4
SPECIAL_TOKEN_LIST = [SPECIAL_TOKENS[role][0] for role in _SPECIAL_ORDER]


def _group_documents(shards: list[Path]) -> Iterator[str]:
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line).get("text", "").strip()
                if text:
                    yield text


def _iter_corpus_text(corpus_dir: Path, sample_tokens: int | None) -> Iterator[str]:
    """Round-robin documents across sources, capped by a whitespace-token budget.

    Interleaving keeps both registers represented (books are token-dense and
    would otherwise dominate the BPE statistics); the token budget bounds
    training time/memory — a 16k vocab converges well under 1B tokens.
    """

    groups: dict[str, list[Path]] = {}
    for shard in sorted(corpus_dir.glob("*.jsonl")):
        groups.setdefault(shard.name.split("-")[0], []).append(shard)
    iterators = {prefix: _group_documents(shards) for prefix, shards in groups.items()}

    tokens = 0
    active = list(iterators)
    while active:
        for prefix in list(active):
            try:
                text = next(iterators[prefix])
            except StopIteration:
                active.remove(prefix)
                continue
            yield text
            tokens += text.count(" ") + 1
            if sample_tokens is not None and tokens >= sample_tokens:
                return


def train_tokenizer(
    *, corpus_dir: Path, output_dir: Path, vocab_size: int, sample_tokens: int | None
) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"][0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKEN_LIST,  # listed first -> ids 0..4
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    budget = "all" if sample_tokens is None else f"~{sample_tokens/1e6:.0f}M ws-tokens"
    print(
        f"Training byte-level BPE (vocab {vocab_size}) on {corpus_dir} "
        f"(interleaved sources, budget {budget})…",
        flush=True,
    )
    tokenizer.train_from_iterator(
        _iter_corpus_text(corpus_dir, sample_tokens), trainer=trainer
    )

    cls_id, sep_id = SPECIAL_TOKENS["cls"][1], SPECIAL_TOKENS["sep"][1]
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKENS['cls'][0]} $A {SPECIAL_TOKENS['sep'][0]}",
        pair=(
            f"{SPECIAL_TOKENS['cls'][0]} $A {SPECIAL_TOKENS['sep'][0]} "
            f"$B {SPECIAL_TOKENS['sep'][0]}"
        ),
        special_tokens=[
            (SPECIAL_TOKENS["cls"][0], cls_id),
            (SPECIAL_TOKENS["sep"][0], sep_id),
        ],
    )

    # Verify the fixed special-token ids came out as intended.
    for role in _SPECIAL_ORDER:
        token, expected_id = SPECIAL_TOKENS[role]
        got = tokenizer.token_to_id(token)
        if got != expected_id:
            raise SystemExit(
                f"Special token {token!r} got id {got}, expected {expected_id}."
            )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=SPECIAL_TOKENS["unk"][0],
        cls_token=SPECIAL_TOKENS["cls"][0],
        sep_token=SPECIAL_TOKENS["sep"][0],
        pad_token=SPECIAL_TOKENS["pad"][0],
        mask_token=SPECIAL_TOKENS["mask"][0],
        bos_token=SPECIAL_TOKENS["cls"][0],
        eos_token=SPECIAL_TOKENS["sep"][0],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output_dir)
    print(
        f"Saved fast tokenizer to {output_dir} "
        f"(vocab {fast.vocab_size}). tokenizer.json + configs written.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=500_000_000,
        help="Whitespace-token budget for BPE training, interleaved across "
        "sources (0 = use the entire corpus).",
    )
    arguments = parser.parse_args()
    train_tokenizer(
        corpus_dir=arguments.corpus,
        output_dir=arguments.output,
        vocab_size=arguments.vocab_size,
        sample_tokens=arguments.sample_tokens or None,
    )


if __name__ == "__main__":
    main()
