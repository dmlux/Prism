from pathlib import Path

from prism.data.english import (
    EnglishUdMorphologyDecoder,
    normalize_english_ud_lemma,
)
from prism.data.gutenberg import (
    gutenberg_plain_paragraphs,
    iter_gutenberg_silver_sentences,
    strip_gutenberg_boilerplate,
)
from prism.languages.english.silver_extraction import (
    english_sentence_extraction_policy,
)


def test_english_lemma_normalization_is_identity() -> None:
    # English UD keeps the literal "$" token; unlike Norwegian there is no
    # "$" marker to strip. British/American spellings pass through unchanged.
    assert normalize_english_ud_lemma("$") == "$"
    assert normalize_english_ud_lemma("colour") == "colour"
    assert normalize_english_ud_lemma("organize") == "organize"


def test_english_morphology_decoder_is_identity() -> None:
    decoder = EnglishUdMorphologyDecoder()
    features = {"Number": "Plur", "Tense": "Past", "VerbForm": "Fin"}
    assert decoder("VERB", features) == features
    # A distinct dict is returned, not the same object.
    assert decoder("VERB", features) is not features


_SYNTHETIC_BOOK = """The Project Gutenberg eBook of A Test Book

Metadata, license blather, produced by volunteers. Ignore me.

*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***

CHAPTER ONE

Mr. Smith travelled to the centre of town on Dec. 3. He met Dr.
Jones, who had organised a colourful gathering, e.g. a fair.

*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***

This footer is Project Gutenberg trademark text and must be dropped.
"""


def test_gutenberg_strips_header_and_footer() -> None:
    body = strip_gutenberg_boilerplate(_SYNTHETIC_BOOK)
    assert "license blather" not in body
    assert "trademark text" not in body
    assert "Mr. Smith travelled" in body


def test_gutenberg_reflows_paragraphs_and_skips_headings() -> None:
    paragraphs = list(
        gutenberg_plain_paragraphs(strip_gutenberg_boilerplate(_SYNTHETIC_BOOK))
    )
    # The all-caps "CHAPTER ONE" heading is skipped.
    assert all("CHAPTER ONE" not in paragraph for paragraph in paragraphs)
    # Hard-wrapped lines are reflowed into a single paragraph.
    assert any(
        "Mr. Smith travelled to the centre of town" in paragraph
        for paragraph in paragraphs
    )


def test_gutenberg_silver_sentences_protect_abbreviations(tmp_path: Path) -> None:
    (tmp_path / "pg9999.txt").write_text(_SYNTHETIC_BOOK, encoding="utf-8")
    policy = english_sentence_extraction_policy(maximum_token_count=128)
    sentences = list(
        iter_gutenberg_silver_sentences(
            archive_path=tmp_path,
            extraction_policy=policy,
        )
    )
    assert sentences
    assert sentences[0].document_id == "pg9999"
    joined = [" ".join(sentence.model_input.tokens) for sentence in sentences]
    # Mr., Dr., Dec. and e.g. must not split the sentence.
    assert any(
        "Mr. Smith" in text and "Dr. Jones" in text and "Dec." in text
        for text in joined
    )
