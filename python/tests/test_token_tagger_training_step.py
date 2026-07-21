from types import SimpleNamespace

import torch
from torch import Tensor, nn

from prism.data import TokenTaskTargetBatch
from prism.modeling import (
    TokenizedBatch,
    TokenTagger,
    TokenTaskHeads,
)
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)
from prism.training import (
    SupervisedTokenTaskBatch,
    train_supervised_token_task_step,
)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
        )

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, return_dict

        return SimpleNamespace(
            last_hidden_state=self.embedding(input_ids),
        )


def test_training_step_updates_backbone_and_task_heads() -> None:
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
    backbone = TinyBackbone()
    heads = TokenTaskHeads(
        hidden_size=4,
        schema=schema,
        dropout_probability=0.0,
    )
    model = TokenTagger(
        backbone=backbone,
        heads=heads,
    )
    batch = SupervisedTokenTaskBatch(
        model_inputs=TokenizedBatch(
            input_ids=torch.tensor(
                [[1, 2, 3]],
                dtype=torch.long,
            ),
            attention_mask=torch.tensor(
                [[True, True, True]],
                dtype=torch.bool,
            ),
            first_subword_indices=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            subword_end_indices=torch.tensor(
                [[2]],
                dtype=torch.long,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            morphology_targets=(
                torch.tensor(
                    [[[False, True]]],
                    dtype=torch.bool,
                ),
            ),
            lemma_rule_ids=torch.tensor(
                [[1]],
                dtype=torch.long,
            ),
            lemma_rule_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
            token_mask=torch.tensor(
                [[True]],
                dtype=torch.bool,
            ),
        ),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    previous_backbone = backbone.embedding.weight.detach().clone()
    previous_upos_head = heads.upos_head.projection.weight.detach().clone()

    losses = train_supervised_token_task_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        max_gradient_norm=1.0,
        morphology_schema=schema.morphology,
    )

    assert torch.isfinite(losses.total_loss)
    assert not torch.equal(
        backbone.embedding.weight.detach(),
        previous_backbone,
    )
    assert not torch.equal(
        heads.upos_head.projection.weight.detach(),
        previous_upos_head,
    )
