import pytest
import torch

from prism.data import PretokenizedSentence
from prism.evaluation import (
    TokenFrequencyClass,
    TokenFrequencyProfile,
    normalize_token_form,
)


def _sentence(*tokens: str) -> PretokenizedSentence:
    return PretokenizedSentence(
        tokens=tokens,
        has_space_before=(False, *(True for _ in tokens[1:])),
    )


def test_token_frequency_profile_classifies_normalized_training_forms() -> None:
    profile = TokenFrequencyProfile.from_sentences(
        (
            _sentence("Å", "hund", "katt", "katt"),
            _sentence("a\u030a", "hund", "katt"),
        ),
        rare_max_frequency=2,
    )

    assert normalize_token_form("A\u030a") == "å"
    assert profile.frequency("å") == 2
    assert profile.classify("Å") is TokenFrequencyClass.RARE
    assert profile.classify("HUND") is TokenFrequencyClass.RARE
    assert profile.classify("katt") is TokenFrequencyClass.FREQUENT
    assert profile.classify("rev") is TokenFrequencyClass.OOV


def test_token_frequency_masks_preserve_sentence_and_token_alignment() -> None:
    profile = TokenFrequencyProfile.from_sentences(
        (_sentence("hund", "hund", "katt", "katt", "katt"),),
        rare_max_frequency=2,
    )
    development_sentences = (
        _sentence("Hund", "rev"),
        _sentence("katt", "gaupe", "hund"),
    )

    rare_masks = profile.build_masks(
        development_sentences,
        frequency_class=TokenFrequencyClass.RARE,
    )
    oov_masks = profile.build_masks(
        development_sentences,
        frequency_class=TokenFrequencyClass.OOV,
    )

    assert rare_masks == ((True, False), (False, False, True))
    assert oov_masks == ((False, True), (False, True, False))
    assert all(
        not (is_rare and is_oov)
        for rare_sentence, oov_sentence in zip(rare_masks, oov_masks, strict=True)
        for is_rare, is_oov in zip(rare_sentence, oov_sentence, strict=True)
    )

    batch_masks = profile.build_batch_masks(
        (development_sentences,),
        frequency_class=TokenFrequencyClass.OOV,
    )
    torch.testing.assert_close(
        batch_masks[0],
        torch.tensor(
            [
                [False, True, False],
                [False, True, False],
            ]
        ),
    )


def test_token_frequency_profile_rejects_invalid_policy() -> None:
    with pytest.raises(
        ValueError,
        match="Rare-token maximum frequency must be positive",
    ):
        TokenFrequencyProfile(
            form_counts={"hund": 1},
            rare_max_frequency=0,
        )
