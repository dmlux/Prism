import torch

from vexo.model import CharacterBiLSTMMultiTaskTagger
from vexo.training import evaluate_multitask


def test_evaluate_multitask() -> None:
    model = CharacterBiLSTMMultiTaskTagger(
        vocabulary_size=3,
        character_count=3,
        tag_count=2,
        feature_count=2,
        word_embedding_size=2,
        character_embedding_size=2,
        character_hidden_size=2,
        hidden_size=2,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.feature_output.bias[1] = 1.0

    batch = (
        torch.tensor([[2, 0]]),
        torch.tensor([[[2], [0]]]),
        torch.tensor([[0, -100]]),
        torch.tensor([[1, -100]]),
        torch.tensor([1]),
        torch.tensor([[1, 0]]),
    )

    (
        pos_loss,
        pos_accuracy,
        feature_loss,
        feature_accuracy,
        annotated_accuracy,
    ) = evaluate_multitask(
        model,
        [batch],
        torch.device("cpu"),
        no_feature_id=0,
    )

    assert pos_loss > 0
    assert feature_loss > 0
    assert pos_accuracy == 1.0
    assert feature_accuracy == 1.0
    assert annotated_accuracy == 1.0