from unittest.mock import MagicMock, Mock

import pytest
import torch

from prism.data import (
    PretokenizedSentence,
    SupervisedSentence,
    TokenTargets,
)
from prism.training import iter_supervised_token_task_batches


def test_supervised_token_task_batches_are_built_lazily() -> None:
    encoded = MagicMock()
    encoded_tensors = {
        "input_ids": torch.tensor(
            [[1, 10, 2]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1]],
            dtype=torch.long,
        ),
    }
    encoded.__getitem__.side_effect = encoded_tensors.__getitem__
    encoded.word_ids.return_value = (None, 0, None)

    tokenizer = Mock(return_value=encoded)
    sentence = SupervisedSentence(
        model_input=PretokenizedSentence(
            tokens=("Hus",),
            has_space_before=(False,),
        ),
        targets=(
            TokenTargets(
                upos_id=0,
                morphology=((True, False),),
                lemma_is_annotated=True,
                lemma_rule_id=0,
            ),
        ),
    )

    batches = iter_supervised_token_task_batches(
        tokenizer=tokenizer,
        sentence_batches=(
            (sentence,),
            (sentence,),
        ),
    )

    assert tokenizer.call_count == 0

    first_batch = next(batches)

    assert tokenizer.call_count == 1
    assert first_batch.batch_size == 1

    second_batch = next(batches)

    assert tokenizer.call_count == 2
    assert second_batch.batch_size == 1

    with pytest.raises(StopIteration):
        next(batches)
