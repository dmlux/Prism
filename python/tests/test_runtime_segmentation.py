from prism.data.segmentation import (
    SentenceExtractionPolicy,
    segment_pretokenized_sentences,
)


_POLICY = SentenceExtractionPolicy(
    abbreviation_tokens=frozenset({"f.eks."}),
    minimum_token_count=3,
    maximum_token_count=8,
    minimum_letter_token_ratio=0.5,
)


def test_runtime_segmentation_keeps_fragments_and_headings() -> None:
    text = "KAPITTEL 1\nHan gjekk heim.\nog so vidare"

    sentences = tuple(segment_pretokenized_sentences(text, _POLICY))

    assert tuple(sentence.tokens for sentence in sentences) == (
        ("KAPITTEL", "1"),
        ("Han", "gjekk", "heim", "."),
        ("og", "so", "vidare"),
    )


def test_runtime_segmentation_chunks_long_sentences_without_loss() -> None:
    words = tuple(f"ord{index}" for index in range(19))
    text = " ".join(words) + "."

    sentences = tuple(segment_pretokenized_sentences(text, _POLICY))

    assert len(sentences) == 3
    assert tuple(len(sentence.tokens) for sentence in sentences) == (8, 8, 4)
    recovered = tuple(token for sentence in sentences for token in sentence.tokens)
    assert recovered == words + (".",)
    assert all(sentence.has_space_before[0] is False for sentence in sentences)


def test_runtime_segmentation_restores_missing_sentence_spaces() -> None:
    text = "De begynte å gå.De gikk fort.«Noe nytt?» spurte han om f.eks.Dette."

    sentences = tuple(segment_pretokenized_sentences(text, _POLICY))

    assert tuple(sentence.tokens for sentence in sentences) == (
        ("De", "begynte", "å", "gå", "."),
        ("De", "gikk", "fort", "."),
        ("«", "Noe", "nytt", "?", "»", "spurte", "han", "om"),
        ("f.eks.", "Dette", "."),
    )


def test_runtime_segmentation_protects_abbreviations_and_ordinals() -> None:
    text = "Vi feirar 17. mai med f.eks. kake. Det er fint."

    sentences = tuple(segment_pretokenized_sentences(text, _POLICY))

    assert tuple(sentence.tokens for sentence in sentences) == (
        ("Vi", "feirar", "17.", "mai", "med", "f.eks.", "kake", "."),
        ("Det", "er", "fint", "."),
    )
