from types import SimpleNamespace

import torch
from torch import Tensor, nn

from prism.modeling import (
    TokenPoolingStrategy,
    TokenTagger,
    TokenTaskHeads,
    TokenizedBatch,
)
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)


class FakeBackbone(nn.Module):
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, return_dict

        hidden_states = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)

        return SimpleNamespace(last_hidden_state=hidden_states)


def test_token_tagger_connects_backbone_alignment_and_task_heads() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(
            version=1,
            labels=("NOUN", "VERB"),
        ),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Sing",),
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
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
    )
    model = TokenTagger(
        backbone=FakeBackbone(),
        heads=heads,
        pooling_strategy=TokenPoolingStrategy.MEAN,
    )
    batch = TokenizedBatch(
        input_ids=torch.tensor([[101, 11, 12, 102]], dtype=torch.long),
        attention_mask=torch.tensor([[True, True, True, True]]),
        first_subword_indices=torch.tensor([[1, 2]], dtype=torch.long),
        subword_end_indices=torch.tensor([[2, 3]], dtype=torch.long),
        token_mask=torch.tensor([[True, True]]),
    )

    logits = model(batch)
    task_hidden_states = model.encode_task_hidden_states(batch)
    logits_from_exposed_boundary = model.heads.classify_hidden_states(
        task_hidden_states,
    )

    assert logits.upos_logits.shape == (1, 2, 2)
    assert logits.morphology_logits[0].shape == (1, 2, 2)
    assert logits.lemma_rule_logits.shape == (1, 2, 2)
    assert task_hidden_states.morphology.shape == (1, 2, 4)
    torch.testing.assert_close(
        logits_from_exposed_boundary.upos_logits,
        logits.upos_logits,
    )
    torch.testing.assert_close(
        logits_from_exposed_boundary.morphology_logits[0],
        logits.morphology_logits[0],
    )
    torch.testing.assert_close(
        logits_from_exposed_boundary.lemma_rule_logits,
        logits.lemma_rule_logits,
    )
