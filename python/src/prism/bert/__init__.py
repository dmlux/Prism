"""PrismBERT — building Prism's own quant-friendly per-language encoder backbones.

This package owns everything for pretraining a PrismBERT backbone from scratch,
separate from the tagger pipeline that consumes it:

* :mod:`prism.bert.corpus` — download a legally-clean, commercial-use-safe
  raw-text corpus (openly licensed, share-alike-compatible with the CC BY-SA 4.0
  model-weight release) and write it as plain JSONL shards with pinned
  provenance;
* (planned) tokenizer training, the MLM pretraining loop (Apple-Silicon / MPS
  friendly, via the canonical LTG GPT-BERT model), and conversion of the
  pretrained checkpoint into a ``transformers``-loadable backbone the existing
  ``prism.modeling`` / ``prism.languages`` pipeline can distil a tagger onto.

The architecture blueprint and staged plan live in ``docs/PRISMBERT.md``.
"""
