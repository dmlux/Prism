import torch

from prism.modeling import TokenizedBatch


def test_tokenized_batch_exposes_input_shapes() -> None:
    batch = TokenizedBatch(
        input_ids=torch.tensor(
            [
                [1, 101, 102, 2],
                [1, 201, 2, 3],
            ],
            dtype=torch.long,
        ),
        attention_mask=torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
            ],
            dtype=torch.bool,
        ),
        first_subword_indices=torch.tensor(
            [
                [1, 2],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        token_mask=torch.tensor(
            [
                [True, True],
                [True, False],
            ],
            dtype=torch.bool,
        ),
    )

    assert batch.batch_size == 2
    assert batch.max_subword_count == 4
    assert batch.max_token_count == 2
