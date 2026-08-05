"""Regenerate a subword-parity fixture for a checked-in example text.

The native byte-level BPE implementations (C++ and Swift) are parity-tested
against the Python reference pipeline: this tool segments a raw text with the
Norwegian runtime segmentation policy and records, per sentence, the subword
IDs the reference Hugging Face tokenizer produces for the artifact's
``vocabulary.json``. The native test suites replay the text through their own
segmentation and BPE and must reproduce these IDs exactly.

Run from the repository root, for example:

    python -m prism.tools.subword_parity_fixture \
      --text data/examples/skarvholmen-bokmaal.txt \
      --vocabulary models/prism-no-0.2.3/vocabulary.json \
      --output data/examples/skarvholmen-bokmaal-subword-parity.json
"""

import argparse
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from prism.data.segmentation import segment_pretokenized_sentences
from prism.languages.norwegian.silver_extraction import (
    norwegian_sentence_extraction_policy,
)
from prism.modeling.tokenizers import prepare_pretokenized_words


def build_fixture(
    *,
    text: str,
    vocabulary_path: Path,
    maximum_token_count: int,
) -> dict[str, object]:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(vocabulary_path))
    policy = norwegian_sentence_extraction_policy(
        maximum_token_count=maximum_token_count,
    )

    sentence_input_ids: list[list[int]] = []
    token_count = 0
    for sentence in segment_pretokenized_sentences(text, policy):
        words = prepare_pretokenized_words(
            tokens=sentence.tokens,
            has_space_before=sentence.has_space_before,
        )
        encoded = tokenizer([list(words)], is_split_into_words=True)
        sentence_input_ids.append(list(encoded["input_ids"][0]))
        token_count += len(sentence.tokens)

    return {
        "policy_version": "prism-runtime-segmentation-v1",
        "maximum_token_count": maximum_token_count,
        "sentence_count": len(sentence_input_ids),
        "token_count": token_count,
        "sentence_input_ids": sentence_input_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-token-count", type=int, default=128)
    arguments = parser.parse_args()

    fixture = build_fixture(
        text=arguments.text.read_text(encoding="utf-8"),
        vocabulary_path=arguments.vocabulary,
        maximum_token_count=arguments.maximum_token_count,
    )
    arguments.output.write_text(
        json.dumps(fixture, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{arguments.output}: {fixture['sentence_count']} sentences, "
        f"{fixture['token_count']} tokens"
    )


if __name__ == "__main__":
    main()
