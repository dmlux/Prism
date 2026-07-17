from prism.data import PretokenizedSentence


def test_pretokenized_sentence_preserves_tokens_and_spacing() -> None:
    sentence = PretokenizedSentence(
        tokens=("Jeg", "så", "filmen", "."),
        has_space_before=(False, True, True, False),
    )

    assert sentence.tokens == ("Jeg", "så", "filmen", ".")
    assert sentence.has_space_before == (False, True, True, False)
