import torch

from prism.modeling import TokenTaskHeads
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)


def test_token_task_heads_create_logits_from_schema() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(
            version=1,
            labels=("ADJ", "NOUN", "VERB"),
        ),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=False,
                ),
                MorphologyFeatureSchema(
                    name="Tense",
                    values=("Past", "Pres"),
                    allows_multiple_values=False,
                ),
            ),
        ),
        lemma_rules=LemmaRuleSchema(
            version=1,
            rules=(
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=0,
                    prefix_addition="",
                    suffix_addition="",
                ),
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=1,
                    prefix_addition="",
                    suffix_addition="",
                ),
            ),
        ),
    )
    heads = TokenTaskHeads(
        hidden_size=192,
        schema=schema,
        dropout_probability=0.1,
    )

    logits = heads(torch.randn((2, 4, 192)))

    assert logits.upos_logits.shape == (2, 4, 3)
    assert tuple(
        feature_logits.shape for feature_logits in logits.morphology_logits
    ) == (
        (2, 4, 3),
        (2, 4, 3),
    )
    assert logits.lemma_rule_logits.shape == (2, 4, 2)
