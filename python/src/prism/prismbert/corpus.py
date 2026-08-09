"""Legally-clean English pretraining corpus for PrismBERT.

Streams openly-licensed sources and writes them as plain JSONL shards plus a
pinned provenance manifest. All sources are commercial-use-safe and
share-alike-compatible with releasing the resulting model weights under
CC BY-SA 4.0 (Prism's strictest license):

* English Wikipedia — ``wikimedia/wikipedia`` config ``20231101.en``
  (CC BY-SA 3.0 + GFDL; one-way upgradeable to CC BY-SA 4.0).
* Project Gutenberg — ``deepmind/pg19`` (public-domain books published before
  1919; Apache-2.0 dataset packaging).

CommonCrawl-derived corpora (FineWeb / C4 / OSCAR / Dolma) are deliberately
EXCLUDED: they are scraped web text with unclear/contested copyright status,
incompatible with a commercial-use guarantee.

Run (writes to data/pretraining/en/, gitignored):
    PYTHONPATH=python/src .venv/bin/python -m prism.prismbert.corpus --language en
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


@dataclass(frozen=True, slots=True)
class CorpusSource:
    key: str
    dataset_id: str
    config: str | None
    split: str
    text_field: str
    license_id: str


EN_SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        key="wikipedia",
        dataset_id="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
        license_id="CC-BY-SA-3.0 / GFDL",
    ),
    CorpusSource(
        key="gutenberg",
        dataset_id="deepmind/pg19",
        config=None,
        split="train",
        text_field="text",
        license_id="public-domain (pre-1919 books; Apache-2.0 packaging)",
    ),
)

SOURCES_BY_LANGUAGE: dict[str, tuple[CorpusSource, ...]] = {"en": EN_SOURCES}


def _resolve_revision(dataset_id: str) -> str:
    """Pin the exact dataset commit for a reproducible provenance chain."""

    return HfApi().dataset_info(dataset_id).sha


def download_corpus(
    *,
    language: str,
    output_root: Path,
    docs_per_shard: int,
    max_docs_per_source: int | None,
) -> None:
    sources = SOURCES_BY_LANGUAGE.get(language)
    if sources is None:
        raise SystemExit(f"No corpus sources registered for language {language!r}.")

    out = output_root / language
    out.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, object] = {
        "language": language,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [],
        "excluded_note": (
            "CommonCrawl-derived corpora excluded for commercial license safety."
        ),
    }

    for source in sources:
        revision = _resolve_revision(source.dataset_id)
        print(
            f"[{source.key}] {source.dataset_id} "
            f"({source.config or 'default'}) rev={revision[:12]} — streaming…",
            flush=True,
        )
        dataset = load_dataset(
            source.dataset_id,
            source.config,
            split=source.split,
            streaming=True,
            revision=revision,
        )

        shard_index = 0
        documents_in_shard = 0
        documents = 0
        whitespace_tokens = 0
        handle = (out / f"{source.key}-{shard_index:05d}.jsonl").open(
            "w", encoding="utf-8"
        )
        try:
            for example in dataset:
                text = (example.get(source.text_field) or "").strip()
                if not text:
                    continue
                handle.write(
                    json.dumps({"text": text, "source": source.key}, ensure_ascii=False)
                    + "\n"
                )
                documents += 1
                documents_in_shard += 1
                whitespace_tokens += text.count(" ") + 1
                if documents_in_shard >= docs_per_shard:
                    handle.close()
                    shard_index += 1
                    documents_in_shard = 0
                    handle = (out / f"{source.key}-{shard_index:05d}.jsonl").open(
                        "w", encoding="utf-8"
                    )
                    print(
                        f"[{source.key}] {documents:,} docs, "
                        f"~{whitespace_tokens / 1e9:.2f}B ws-tokens",
                        flush=True,
                    )
                if max_docs_per_source is not None and documents >= max_docs_per_source:
                    break
        finally:
            handle.close()

        provenance["sources"].append(
            {
                "key": source.key,
                "dataset_id": source.dataset_id,
                "revision": revision,
                "config": source.config,
                "split": source.split,
                "license": source.license_id,
                "documents": documents,
                "approx_whitespace_tokens": whitespace_tokens,
            }
        )
        print(
            f"[{source.key}] DONE: {documents:,} docs, "
            f"~{whitespace_tokens / 1e9:.2f}B ws-tokens",
            flush=True,
        )

    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total = sum(s["approx_whitespace_tokens"] for s in provenance["sources"])
    print(
        f"\nCorpus ready at {out} — {len(provenance['sources'])} sources, "
        f"~{total / 1e9:.2f}B whitespace-tokens total. Provenance: provenance.json",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--docs-per-shard", type=int, default=100_000)
    parser.add_argument(
        "--max-docs-per-source",
        type=int,
        default=None,
        help="Cap documents per source (smoke runs); default streams everything.",
    )
    arguments = parser.parse_args()
    download_corpus(
        language=arguments.language,
        output_root=arguments.output_root,
        docs_per_shard=arguments.docs_per_shard,
        max_docs_per_source=arguments.max_docs_per_source,
    )


if __name__ == "__main__":
    main()
