from typing import NamedTuple

import pytest
import torch
from torch import Tensor, nn

from prism.exporting import (
    CharacterAwareTokenTaggerExportAdapter,
    TokenTaggerExportAdapter,
)
from prism.modeling import (
    CharacterCnnTokenEncoder,
    MorphologyLogitCorrection,
    MorphologyAgreementRefinerSpec,
    MorphologyBundleCandidate,
    MorphologyBundleRerankerSpec,
    TokenPoolingStrategy,
    TokenTagger,
    TokenTaskHeadArchitecture,
    TokenTaskHeads,
    TokenTaskLogits,
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


class TinyExportTokenTagger(nn.Module):
    def forward(self, batch: TokenizedBatch) -> TokenTaskLogits:
        hidden_states = batch.input_ids.to(dtype=torch.float32).unsqueeze(-1)

        return TokenTaskLogits(
            upos_logits=torch.cat((hidden_states, hidden_states + 1.0), dim=-1),
            morphology_logits=(
                torch.cat((hidden_states + 2.0, hidden_states + 3.0), dim=-1),
            ),
            lemma_rule_logits=torch.cat(
                (hidden_states + 4.0, hidden_states + 5.0),
                dim=-1,
            ),
        )


class TinyBackboneOutput(NamedTuple):
    last_hidden_state: Tensor


class TinyExportBackbone(nn.Module):
    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_dict: bool,
    ) -> TinyBackboneOutput:
        del attention_mask, return_dict

        hidden_states = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)

        return TinyBackboneOutput(last_hidden_state=hidden_states)


def test_token_tagger_export_adapter_has_flat_tensor_contract() -> None:
    adapter = TokenTaggerExportAdapter(model=TinyExportTokenTagger())
    inputs = (
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[True, True, True]]),
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[2, 3]], dtype=torch.long),
        torch.tensor([[True, True]]),
    )

    eager_outputs = adapter(*inputs)
    exported = torch.export.export(adapter, inputs, strict=True)
    exported_outputs = exported.module()(*inputs)

    assert len(exported_outputs) == 3
    for eager, exported_output in zip(
        eager_outputs,
        exported_outputs,
        strict=True,
    ):
        torch.testing.assert_close(exported_output, eager)


def test_token_tagger_export_embeds_morphology_logit_correction() -> None:
    correction = MorphologyLogitCorrection(
        strength=1.0,
        weights=(torch.tensor([1.0, 4.0]),),
    )
    adapter = TokenTaggerExportAdapter(
        model=TinyExportTokenTagger(),
        morphology_logit_correction=correction,
    )
    inputs = (
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[True, True, True]]),
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[2, 3]], dtype=torch.long),
        torch.tensor([[True, True]]),
    )

    raw_outputs = TokenTaggerExportAdapter(model=TinyExportTokenTagger())(*inputs)
    corrected_outputs = adapter(*inputs)
    exported = torch.export.export(adapter, inputs, strict=True)
    exported_outputs = exported.module()(*inputs)

    assert "morphology_logit_correction.offset_0" in adapter.state_dict()
    torch.testing.assert_close(corrected_outputs[0], raw_outputs[0])
    torch.testing.assert_close(
        corrected_outputs[1],
        raw_outputs[1] - correction.weights[0].log(),
    )
    torch.testing.assert_close(corrected_outputs[2], raw_outputs[2])
    for eager, exported_output in zip(
        corrected_outputs,
        exported_outputs,
        strict=True,
    ):
        torch.testing.assert_close(exported_output, eager)


def test_character_aware_token_tagger_has_strict_export_parity() -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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
    model = TokenTagger(
        backbone=TinyExportBackbone(),
        heads=TokenTaskHeads(
            hidden_size=4,
            schema=schema,
            dropout_probability=0.0,
            architecture=(
                TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY_CHARACTER_CNN
            ),
            morphology_bundle_reranker_spec=MorphologyBundleRerankerSpec(
                maximum_candidates_per_upos=1,
                candidates=(
                    MorphologyBundleCandidate(
                        upos_id=0,
                        morphology=((False, True, False),),
                        training_count=2,
                    ),
                    MorphologyBundleCandidate(
                        upos_id=1,
                        morphology=((True, False, False),),
                        training_count=1,
                    ),
                ),
            ),
            morphology_agreement_refiner_spec=MorphologyAgreementRefinerSpec(
                window_radius=3,
                bottleneck_size=4,
                target_feature_names=("Number",),
            ),
        ),
        pooling_strategy=TokenPoolingStrategy.MEAN,
        character_encoder=CharacterCnnTokenEncoder(
            vocabulary_size=16,
            hidden_size=4,
            embedding_size=4,
        ),
    )
    model.eval()
    correction = MorphologyLogitCorrection(
        strength=1.0,
        weights=(torch.tensor([1.0, 4.0]),),
    )
    adapter = CharacterAwareTokenTaggerExportAdapter(
        model=model,
        morphology_logit_correction=correction,
    )
    inputs = (
        torch.tensor([[101, 11, 12, 102]], dtype=torch.long),
        torch.tensor([[True, True, True, True]]),
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[2, 3]], dtype=torch.long),
        torch.tensor([[True, True]]),
        torch.tensor([[[2, 5, 6, 3], [2, 7, 8, 3]]], dtype=torch.long),
        torch.tensor([[[True, True, True, True], [True, True, True, True]]]),
    )

    eager_outputs = adapter(*inputs)
    exported_outputs = torch.export.export(
        adapter,
        inputs,
        strict=True,
    ).module()(*inputs)

    assert len(exported_outputs) == 3
    assert "morphology_logit_correction.offset_0" in adapter.state_dict()
    for eager, exported_output in zip(
        eager_outputs,
        exported_outputs,
        strict=True,
    ):
        torch.testing.assert_close(exported_output, eager)


@pytest.mark.parametrize(
    "architecture",
    (
        TokenTaskHeadArchitecture.WIDE_SHARED_MLP_TASK_ADAPTERS,
        TokenTaskHeadArchitecture.WIDE_SHARED_MLP_STRUCTURED_MORPHOLOGY,
    ),
)
def test_extended_token_tagger_has_strict_export_parity(
    architecture: TokenTaskHeadArchitecture,
) -> None:
    schema = TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=True,
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
    model = TokenTagger(
        backbone=TinyExportBackbone(),
        heads=TokenTaskHeads(
            hidden_size=4,
            schema=schema,
            dropout_probability=0.0,
            architecture=architecture,
        ),
        pooling_strategy=TokenPoolingStrategy.MEAN,
    )
    model.eval()
    adapter = TokenTaggerExportAdapter(model=model)
    inputs = (
        torch.tensor([[101, 11, 12, 102]], dtype=torch.long),
        torch.tensor([[True, True, True, True]]),
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[2, 3]], dtype=torch.long),
        torch.tensor([[True, True]]),
    )

    eager_outputs = adapter(*inputs)
    exported_outputs = torch.export.export(
        adapter,
        inputs,
        strict=True,
    ).module()(*inputs)

    assert len(exported_outputs) == 3
    for eager, exported_output in zip(
        eager_outputs,
        exported_outputs,
        strict=True,
    ):
        torch.testing.assert_close(exported_output, eager)
