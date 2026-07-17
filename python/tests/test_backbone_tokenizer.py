from unittest.mock import Mock, patch

from prism.languages.norwegian import NORBERT4_XSMALL_BACKBONE
from prism.modeling import (
    load_backbone_tokenizer,
    prepare_pretokenized_words,
)


def test_load_backbone_tokenizer_uses_pinned_spec() -> None:
    tokenizer = Mock(is_fast=True)

    with patch(
        "prism.modeling.tokenizers.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as from_pretrained:
        loaded_tokenizer = load_backbone_tokenizer(NORBERT4_XSMALL_BACKBONE)

    assert loaded_tokenizer is tokenizer
    from_pretrained.assert_called_once_with(
        "ltg/norbert4-xsmall",
        revision=("7483327d36a2daa5dbe936c68aa277149c6f9632"),
        trust_remote_code=True,
    )


def test_prepare_pretokenized_words_preserves_spacing() -> None:
    words = prepare_pretokenized_words(
        tokens=("Jeg", "så", "filmen", "."),
        has_space_before=(False, True, True, False),
    )

    assert words == ("Jeg", " så", " filmen", ".")
