import pytest

from prism.data import SentenceExtractionPolicy, extract_pretokenized_sentences
from prism.languages.norwegian.silver_extraction import (
    norwegian_sentence_extraction_policy,
)


def _policy(**overrides: object) -> SentenceExtractionPolicy:
    values: dict[str, object] = {
        "abbreviation_tokens": frozenset({"f.eks.", "nr.", "kl."}),
        "minimum_token_count": 3,
        "maximum_token_count": 64,
        "minimum_letter_token_ratio": 0.5,
    }
    values.update(overrides)
    return SentenceExtractionPolicy(**values)  # type: ignore[arg-type]


def test_extraction_merges_wrapped_lines_and_splits_sentences() -> None:
    text = (
        "Eventuelt forfall må meldast til utvalssekretæren\n"
        "straks. Varamedlemer møter berre etter særskild innkalling.\n"
    )

    sentences = tuple(extract_pretokenized_sentences(text, _policy()))

    assert tuple(sentence.tokens for sentence in sentences) == (
        (
            "Eventuelt",
            "forfall",
            "må",
            "meldast",
            "til",
            "utvalssekretæren",
            "straks",
            ".",
        ),
        ("Varamedlemer", "møter", "berre", "etter", "særskild", "innkalling", "."),
    )
    assert sentences[0].has_space_before == (
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    )


def test_extraction_repairs_ocr_hyphenation() -> None:
    text = "Kommunen vedtok eit nytt klima-\nbudsjett i fjor.\n"

    sentences = tuple(extract_pretokenized_sentences(text, _policy()))

    assert tuple(sentences[0].tokens) == (
        "Kommunen",
        "vedtok",
        "eit",
        "nytt",
        "klimabudsjett",
        "i",
        "fjor",
        ".",
    )


def test_extraction_protects_abbreviations_and_ordinals() -> None:
    text = (
        "Møtet gjeld f.eks. Bergen kommune sitt budsjett.\n"
        "Feiringa av 17. Mai vart utsett til kl. 12.\n"
    )

    sentences = tuple(extract_pretokenized_sentences(text, _policy()))

    assert tuple(tuple(sentence.tokens) for sentence in sentences) == (
        (
            "Møtet",
            "gjeld",
            "f.eks.",
            "Bergen",
            "kommune",
            "sitt",
            "budsjett",
            ".",
        ),
        ("Feiringa", "av", "17.", "Mai", "vart", "utsett", "til", "kl.", "12", "."),
    )


def test_extraction_discards_headers_tables_and_fragments() -> None:
    text = (
        "MØTEINNKALLING\n"
        "Saksnr.\n"
        "17/758\n"
        "Tid: 10:00\n"
        "\n"
        "utan stor forbokstav i starten.\n"
        "Kort.\n"
        "Denne setninga er derimot heilt vanleg prosa.\n"
    )

    sentences = tuple(extract_pretokenized_sentences(text, _policy()))

    assert tuple(tuple(sentence.tokens) for sentence in sentences) == (
        ("Denne", "setninga", "er", "derimot", "heilt", "vanleg", "prosa", "."),
    )


def test_extraction_keeps_numbers_case_ids_and_attached_punctuation() -> None:
    text = "Saka 2020/42652-5 vart handsama 07.02.2017, sjå nr. 4.\n"

    sentences = tuple(extract_pretokenized_sentences(text, _policy()))

    assert tuple(sentences[0].tokens) == (
        "Saka",
        "2020/42652-5",
        "vart",
        "handsama",
        "07.02.2017",
        ",",
        "sjå",
        "nr.",
        "4",
        ".",
    )
    comma_index = sentences[0].tokens.index(",")
    assert sentences[0].has_space_before[comma_index] is False


def test_extraction_enforces_token_count_and_letter_ratio() -> None:
    long_policy = _policy(maximum_token_count=5)
    too_long = "Denne setninga har altfor mange tokens for grensa.\n"
    numeric = "Ei 1 2 3 4 5 6 7 8 9 rekkje.\n"

    assert tuple(extract_pretokenized_sentences(too_long, long_policy)) == ()
    assert tuple(extract_pretokenized_sentences(numeric, _policy())) == ()


def test_extraction_discards_ocr_dot_leaders_and_quoted_fragments() -> None:
    text = (
        "Vatn og avlaup er viktige tema..\n"
        "Innhaldsliste med mange punkt....\n"
        '"... ikkje rekna som del av miljøet på Elvebakken.\n'
        "«Dette sitatet er derimot komplett og gyldig.»\n"
        "Han nølte litt før svaret kom...\n"
    )

    sentences = tuple(extract_pretokenized_sentences(text, _policy()))

    reconstructed = tuple(" ".join(sentence.tokens) for sentence in sentences)
    assert reconstructed == (
        "« Dette sitatet er derimot komplett og gyldig . »",
        "Han nølte litt før svaret kom . . .",
    )


def test_norwegian_policy_rejects_invalid_abbreviations() -> None:
    with pytest.raises(ValueError, match="Abbreviation tokens"):
        SentenceExtractionPolicy(
            abbreviation_tokens=frozenset({"Nr."}),
            minimum_token_count=1,
            maximum_token_count=2,
            minimum_letter_token_ratio=0.5,
        )

    policy = norwegian_sentence_extraction_policy(maximum_token_count=128)
    assert "bl.a." in policy.abbreviation_tokens
    assert policy.minimum_token_count == 4
