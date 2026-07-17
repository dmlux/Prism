from prism.data import (
    PretokenizedSentence,
    SupervisedSentence,
    TokenTargets,
)
from prism.training import build_supervised_sentence_batches


def _sentence(token: str) -> SupervisedSentence:
    return SupervisedSentence(
        model_input=PretokenizedSentence(
            tokens=(token,),
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


def test_supervised_sentence_batches_are_reproducible() -> None:
    sentences = tuple(_sentence(f"token-{index}") for index in range(5))

    batches = build_supervised_sentence_batches(
        sentences=sentences,
        batch_size=2,
        random_seed=42,
        epoch_index=0,
    )
    repeated_batches = build_supervised_sentence_batches(
        sentences=sentences,
        batch_size=2,
        random_seed=42,
        epoch_index=0,
    )

    assert tuple(len(batch) for batch in batches) == (2, 2, 1)
    assert batches == repeated_batches

    batched_tokens = {
        sentence.model_input.tokens[0] for batch in batches for sentence in batch
    }

    assert batched_tokens == {
        "token-0",
        "token-1",
        "token-2",
        "token-3",
        "token-4",
    }
