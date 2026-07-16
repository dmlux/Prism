import pytest

from prism.data.norwegian_bokmaal import (
    normalize_norwegian_bokmaal_ud_lemma,
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
def test_normalize_norwegian_bokmaal_ud_lemma_removes_marker(
    raw_lemma: str,
    expected_lemma: str,
) -> None:
    assert normalize_norwegian_bokmaal_ud_lemma(raw_lemma) == expected_lemma
