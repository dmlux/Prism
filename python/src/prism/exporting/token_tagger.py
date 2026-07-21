from torch import Tensor, nn

from prism.modeling import CharacterTokenBatch, TokenizedBatch


class TokenTaggerExportAdapter(nn.Module):
    """Expose the token tagger through a flat tensor-only export contract."""

    def __init__(self, *, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        first_subword_indices: Tensor,
        subword_end_indices: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        output = self.model(
            TokenizedBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_indices=first_subword_indices,
                subword_end_indices=subword_end_indices,
                token_mask=token_mask,
            )
        )

        return (
            output.upos_logits,
            *output.morphology_logits,
            output.lemma_rule_logits,
        )


class CharacterAwareTokenTaggerExportAdapter(nn.Module):
    """Expose a character-aware token tagger through flat tensor inputs."""

    def __init__(self, *, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        first_subword_indices: Tensor,
        subword_end_indices: Tensor,
        token_mask: Tensor,
        character_ids: Tensor,
        character_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        output = self.model(
            TokenizedBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_indices=first_subword_indices,
                subword_end_indices=subword_end_indices,
                token_mask=token_mask,
            ),
            CharacterTokenBatch(
                character_ids=character_ids,
                character_mask=character_mask,
                token_mask=token_mask,
            ),
        )

        return (
            output.upos_logits,
            *output.morphology_logits,
            output.lemma_rule_logits,
        )
