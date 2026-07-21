import torch

from prism.data import (
    PretokenizedSentence,
    SupervisedSentence,
    TokenTargets,
    build_token_task_target_batch,
)


def test_build_token_task_target_batch_pads_sentences() -> None:
    sentences = (
        SupervisedSentence(
            model_input=PretokenizedSentence(
                tokens=("Hus", "står"),
                has_space_before=(False, True),
            ),
            targets=(
                TokenTargets(
                    upos_id=1,
                    morphology=((False, True),),
                    lemma_is_annotated=True,
                    lemma_rule_id=2,
                ),
                TokenTargets(
                    upos_id=0,
                    morphology=((True, False),),
                    lemma_is_annotated=False,
                    lemma_rule_id=None,
                ),
            ),
        ),
        SupervisedSentence(
            model_input=PretokenizedSentence(
                tokens=("Ukjent",),
                has_space_before=(False,),
            ),
            targets=(
                TokenTargets(
                    upos_id=2,
                    morphology=((False, True),),
                    lemma_is_annotated=True,
                    lemma_rule_id=None,
                ),
            ),
        ),
    )

    batch = build_token_task_target_batch(sentences)

    torch.testing.assert_close(
        batch.upos_ids,
        torch.tensor(
            [
                [1, 0],
                [2, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        batch.morphology_targets[0],
        torch.tensor(
            [
                [[False, True], [True, False]],
                [[False, True], [False, False]],
            ],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        batch.lemma_rule_ids,
        torch.tensor(
            [
                [2, 0],
                [0, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        batch.lemma_rule_mask,
        torch.tensor(
            [
                [True, False],
                [False, False],
            ],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        batch.lemma_annotation_mask,
        torch.tensor(
            [
                [True, False],
                [True, False],
            ],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        batch.token_mask,
        torch.tensor(
            [
                [True, True],
                [True, False],
            ],
            dtype=torch.bool,
        ),
    )
