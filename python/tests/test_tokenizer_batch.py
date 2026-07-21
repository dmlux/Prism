from unittest.mock import MagicMock, Mock

import torch

from prism.data import PretokenizedSentence
from prism.modeling import tokenize_pretokenized_sentences


def test_tokenize_pretokenized_sentences_builds_model_batch() -> None:
    encoded = MagicMock()
    encoded_tensors = {
        "input_ids": torch.tensor(
            [[1, 10, 20, 30, 31, 40]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
        ),
    }
    encoded.__getitem__.side_effect = encoded_tensors.__getitem__
    encoded.word_ids.return_value = (None, 0, 1, 2, 2, 3)

    tokenizer = Mock(return_value=encoded)

    batch = tokenize_pretokenized_sentences(
        tokenizer=tokenizer,
        sentences=(
            PretokenizedSentence(
                tokens=("Jeg", "så", "filmen", "."),
                has_space_before=(False, True, True, False),
            ),
        ),
    )

    tokenizer.assert_called_once_with(
        [["Jeg", " så", " filmen", "."]],
        is_split_into_words=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded.word_ids.assert_called_once_with(batch_index=0)

    assert torch.equal(
        batch.input_ids,
        encoded_tensors["input_ids"],
    )
    assert torch.equal(
        batch.attention_mask,
        torch.tensor(
            [[True, True, True, True, True, True]],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        batch.first_subword_indices,
        torch.tensor(
            [[1, 2, 3, 5]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        batch.subword_end_indices,
        torch.tensor(
            [[2, 3, 5, 6]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        batch.token_mask,
        torch.tensor(
            [[True, True, True, True]],
            dtype=torch.bool,
        ),
    )
