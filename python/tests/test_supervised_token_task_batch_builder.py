from unittest.mock import MagicMock, Mock

import torch

from prism.data import (
    PretokenizedSentence,
    SupervisedSentence,
    TokenTargets,
)
from prism.training import build_supervised_token_task_batch
from prism.schema import build_character_vocabulary_schema


def test_build_supervised_token_task_batch_connects_tokenizer_and_targets() -> None:
    encoded = MagicMock()
    encoded_tensors = {
        "input_ids": torch.tensor(
            [[1, 10, 20, 2]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1]],
            dtype=torch.long,
        ),
    }
    encoded.__getitem__.side_effect = encoded_tensors.__getitem__
    encoded.word_ids.return_value = (None, 0, 1, None)

    tokenizer = Mock(return_value=encoded)
    sentence = SupervisedSentence(
        model_input=PretokenizedSentence(
            tokens=("Hus", "står"),
            has_space_before=(False, True),
        ),
        targets=(
            TokenTargets(
                upos_id=0,
                morphology=((True, False),),
                lemma_is_annotated=True,
                lemma_rule_id=0,
            ),
            TokenTargets(
                upos_id=1,
                morphology=((False, True),),
                lemma_is_annotated=True,
                lemma_rule_id=1,
            ),
        ),
    )

    batch = build_supervised_token_task_batch(
        tokenizer=tokenizer,
        sentences=(sentence,),
    )

    tokenizer.assert_called_once_with(
        [["Hus", " står"]],
        is_split_into_words=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    torch.testing.assert_close(
        batch.model_inputs.token_mask,
        batch.targets.token_mask,
    )
    torch.testing.assert_close(
        batch.targets.upos_ids,
        torch.tensor(
            [[0, 1]],
            dtype=torch.long,
        ),
    )

    character_vocabulary = build_character_vocabulary_schema(
        tokens=sentence.model_input.tokens
    )
    character_batch = build_supervised_token_task_batch(
        tokenizer=tokenizer,
        sentences=(sentence,),
        character_vocabulary=character_vocabulary,
        maximum_character_count=8,
    )

    assert character_batch.character_inputs is not None
    torch.testing.assert_close(
        character_batch.character_inputs.token_mask,
        character_batch.targets.token_mask,
    )
