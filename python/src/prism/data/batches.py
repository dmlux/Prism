from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from prism.data.examples import SupervisedSentence


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskTargetBatch:
    upos_ids: Tensor
    morphology_targets: tuple[Tensor, ...]
    lemma_rule_ids: Tensor
    lemma_rule_mask: Tensor
    token_mask: Tensor
    lemma_annotation_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.upos_ids.ndim != 2:
            raise ValueError("UPOS target IDs must have two dimensions.")
        if self.upos_ids.dtype != torch.long:
            raise ValueError("UPOS target IDs must use torch.long.")
        if not self.morphology_targets:
            raise ValueError("Target batch must contain morphology feature targets.")

        token_dimensions = self.upos_ids.shape

        if any(targets.ndim != 3 for targets in self.morphology_targets):
            raise ValueError("Morphology targets must have three dimensions.")
        if any(
            targets.shape[:2] != token_dimensions for targets in self.morphology_targets
        ):
            raise ValueError(
                "Morphology targets must share batch and token dimensions."
            )
        if any(targets.dtype != torch.bool for targets in self.morphology_targets):
            raise ValueError("Morphology targets must use torch.bool.")

        if self.lemma_rule_ids.shape != token_dimensions:
            raise ValueError("Lemma rule IDs must match UPOS target dimensions.")
        if self.lemma_rule_ids.dtype != torch.long:
            raise ValueError("Lemma rule IDs must use torch.long.")

        for mask in (self.lemma_rule_mask, self.token_mask):
            if mask.shape != token_dimensions:
                raise ValueError("Target masks must match UPOS target dimensions.")
            if mask.dtype != torch.bool:
                raise ValueError("Target masks must use torch.bool.")

        if self.lemma_annotation_mask is None:
            object.__setattr__(
                self,
                "lemma_annotation_mask",
                self.lemma_rule_mask,
            )
        else:
            if self.lemma_annotation_mask.shape != token_dimensions:
                raise ValueError(
                    "Lemma annotation mask must match UPOS target dimensions."
                )
            if self.lemma_annotation_mask.dtype != torch.bool:
                raise ValueError("Lemma annotation mask must use torch.bool.")
            if self.lemma_annotation_mask.device != self.token_mask.device:
                raise ValueError(
                    "Lemma annotation mask and token mask must use the same device."
                )
            if (self.lemma_annotation_mask & ~self.token_mask).any().item():
                raise ValueError(
                    "Lemma annotation mask must not select padding tokens."
                )
            if (self.lemma_rule_mask & ~self.lemma_annotation_mask).any().item():
                raise ValueError(
                    "Representable lemma rules must have lemma annotations."
                )

    @property
    def batch_size(self) -> int:
        return self.upos_ids.shape[0]

    @property
    def max_token_count(self) -> int:
        return self.upos_ids.shape[1]

    @property
    def morphology_feature_count(self) -> int:
        return len(self.morphology_targets)

    def to(self, device: torch.device) -> "TokenTaskTargetBatch":
        if self.lemma_annotation_mask is None:
            raise RuntimeError("Lemma annotation mask must be resolved.")

        return TokenTaskTargetBatch(
            upos_ids=self.upos_ids.to(device=device),
            morphology_targets=tuple(
                targets.to(device=device) for targets in self.morphology_targets
            ),
            lemma_rule_ids=self.lemma_rule_ids.to(device=device),
            lemma_rule_mask=self.lemma_rule_mask.to(device=device),
            token_mask=self.token_mask.to(device=device),
            lemma_annotation_mask=self.lemma_annotation_mask.to(device=device),
        )


def build_token_task_target_batch(
    sentences: Sequence[SupervisedSentence],
) -> TokenTaskTargetBatch:
    if not sentences:
        raise ValueError("Target batch must contain sentences.")

    first_target = sentences[0].targets[0]
    morphology_label_counts = tuple(len(labels) for labels in first_target.morphology)
    max_token_count = max(len(sentence.targets) for sentence in sentences)
    batch_size = len(sentences)

    upos_ids = torch.zeros(
        (batch_size, max_token_count),
        dtype=torch.long,
    )
    morphology_targets = tuple(
        torch.zeros((batch_size, max_token_count, label_count), dtype=torch.bool)
        for label_count in morphology_label_counts
    )
    lemma_rule_ids = torch.zeros(
        (batch_size, max_token_count),
        dtype=torch.long,
    )
    lemma_rule_mask = torch.zeros(
        (batch_size, max_token_count),
        dtype=torch.bool,
    )
    lemma_annotation_mask = torch.zeros(
        (batch_size, max_token_count),
        dtype=torch.bool,
    )
    token_mask = torch.zeros(
        (batch_size, max_token_count),
        dtype=torch.bool,
    )

    for sentence_index, sentence in enumerate(sentences):
        for token_index, target in enumerate(sentence.targets):
            if len(target.morphology) != len(morphology_targets):
                raise ValueError(
                    "All targets must contain the same morphology features."
                )

            upos_ids[sentence_index, token_index] = target.upos_id
            token_mask[sentence_index, token_index] = True

            for feature_index, labels in enumerate(target.morphology):
                if len(labels) != morphology_label_counts[feature_index]:
                    raise ValueError("Morphology label counts must be consistent.")

                morphology_targets[feature_index][
                    sentence_index,
                    token_index,
                ] = torch.tensor(
                    labels,
                    dtype=torch.bool,
                )

            if target.lemma_rule_id is not None:
                lemma_rule_ids[
                    sentence_index,
                    token_index,
                ] = target.lemma_rule_id
                lemma_rule_mask[
                    sentence_index,
                    token_index,
                ] = True
            if target.lemma_is_annotated:
                lemma_annotation_mask[sentence_index, token_index] = True

    return TokenTaskTargetBatch(
        upos_ids=upos_ids,
        morphology_targets=morphology_targets,
        lemma_rule_ids=lemma_rule_ids,
        lemma_rule_mask=lemma_rule_mask,
        token_mask=token_mask,
        lemma_annotation_mask=lemma_annotation_mask,
    )
