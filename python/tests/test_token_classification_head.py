import torch

from prism.modeling import TokenClassificationHead


def test_token_classification_head_maps_each_token_to_label_logits() -> None:
    head = TokenClassificationHead(
        hidden_size=192,
        label_count=17,
        dropout_probability=0.0,
    )
    hidden_states = torch.randn((2, 4, 192))

    logits = head(hidden_states)

    assert logits.shape == (2, 4, 17)
