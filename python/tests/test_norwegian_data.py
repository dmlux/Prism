import pytest

from prism.conllu import Token
from prism.data.norwegian import (
    NorwegianUdMorphologyDecoder,
    build_norwegian_ud_lemma_decoder,
    normalize_norwegian_ud_lemma,
)


@pytest.mark.parametrize(
    ("raw_lemma", "expected_lemma"),
    [
        ("hus", "hus"),
        ("$.", "."),
        ('$"', '"'),
        ("$$", "$"),
    ],
)
def test_normalize_norwegian_ud_lemma_removes_marker(
    raw_lemma: str,
    expected_lemma: str,
) -> None:
    assert normalize_norwegian_ud_lemma(raw_lemma) == expected_lemma


def test_norwegian_ud_lemma_decoder_restores_training_marker_convention() -> None:
    decoder = build_norwegian_ud_lemma_decoder(
        (
            (
                Token(text="hus", lemma="hus", upos="NOUN", features={}),
                Token(text="«", lemma='$"', upos="PUNCT", features={}),
            ),
        )
    )

    assert decoder("hus", "hus", "NOUN") == "hus"
    assert decoder("«", '"', "PUNCT") == '$"'


def test_norwegian_ud_morphology_decoder_maps_common_gender_by_upos() -> None:
    decoder = NorwegianUdMorphologyDecoder(language_tag="nb")

    assert decoder(
        "ADJ",
        {"Gender": "Fem,Masc", "Number": "Sing"},
    ) == {"Gender": "Com", "Number": "Sing"}
    assert decoder(
        "PRON",
        {"Gender": "Fem,Masc", "Number": "Sing"},
    ) == {"Gender": "Fem,Masc", "Number": "Sing"}


def test_nynorsk_ud_morphology_decoder_removes_unexpressed_values() -> None:
    decoder = NorwegianUdMorphologyDecoder(language_tag="nn")

    assert decoder(
        "ADJ",
        {
            "Definite": "Def,Ind",
            "Gender": "Fem,Masc",
            "Number": "Plur,Sing",
        },
    ) == {
        "Definite": "Ind",
        "Gender": "Com",
        "Number": "Plur",
    }
