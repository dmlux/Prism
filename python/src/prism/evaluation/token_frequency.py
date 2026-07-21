import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import torch
from torch import Tensor

from prism.data import PretokenizedSentence


class TokenFrequencyClass(StrEnum):
    FREQUENT = "frequent"
    RARE = "rare"
    OOV = "oov"


def normalize_token_form(token: str) -> str:
    if not token:
        raise ValueError("Token form must not be empty.")

    return unicodedata.normalize("NFC", token).casefold()


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenFrequencyProfile:
    form_counts: Mapping[str, int]
    rare_max_frequency: int = 5

    def __post_init__(self) -> None:
        if self.rare_max_frequency <= 0:
            raise ValueError("Rare-token maximum frequency must be positive.")
        if not self.form_counts:
            raise ValueError("Token-frequency profile must contain training forms.")
        if any(not form or count <= 0 for form, count in self.form_counts.items()):
            raise ValueError(
                "Token-frequency profile forms and counts must be positive."
            )

        normalized_counts: Counter[str] = Counter()
        for form, count in self.form_counts.items():
            normalized_counts[normalize_token_form(form)] += count

        object.__setattr__(
            self,
            "form_counts",
            MappingProxyType(dict(normalized_counts)),
        )

    @classmethod
    def from_sentences(
        cls,
        sentences: Iterable[PretokenizedSentence],
        *,
        rare_max_frequency: int = 5,
    ) -> "TokenFrequencyProfile":
        return cls.from_token_sequences(
            (sentence.tokens for sentence in sentences),
            rare_max_frequency=rare_max_frequency,
        )

    @classmethod
    def from_token_sequences(
        cls,
        token_sequences: Iterable[Sequence[str]],
        *,
        rare_max_frequency: int = 5,
    ) -> "TokenFrequencyProfile":
        form_counts = Counter(
            normalize_token_form(token)
            for tokens in token_sequences
            for token in tokens
        )

        return cls(
            form_counts=form_counts,
            rare_max_frequency=rare_max_frequency,
        )

    def frequency(self, token: str) -> int:
        return self.form_counts.get(normalize_token_form(token), 0)

    def classify(self, token: str) -> TokenFrequencyClass:
        frequency = self.frequency(token)
        if frequency == 0:
            return TokenFrequencyClass.OOV
        if frequency <= self.rare_max_frequency:
            return TokenFrequencyClass.RARE
        return TokenFrequencyClass.FREQUENT

    def build_masks(
        self,
        sentences: Sequence[PretokenizedSentence],
        *,
        frequency_class: TokenFrequencyClass,
    ) -> tuple[tuple[bool, ...], ...]:
        if not sentences:
            raise ValueError("Token-frequency masks require sentences.")

        return tuple(
            tuple(self.classify(token) is frequency_class for token in sentence.tokens)
            for sentence in sentences
        )

    def build_batch_masks(
        self,
        sentence_batches: Sequence[Sequence[PretokenizedSentence]],
        *,
        frequency_class: TokenFrequencyClass,
    ) -> tuple[Tensor, ...]:
        if not sentence_batches or any(not batch for batch in sentence_batches):
            raise ValueError("Token-frequency batch masks require non-empty batches.")

        batch_masks: list[Tensor] = []
        for sentence_batch in sentence_batches:
            max_token_count = max(len(sentence.tokens) for sentence in sentence_batch)
            batch_mask = torch.zeros(
                (len(sentence_batch), max_token_count),
                dtype=torch.bool,
            )

            for sentence_index, sentence in enumerate(sentence_batch):
                sentence_mask = tuple(
                    self.classify(token) is frequency_class for token in sentence.tokens
                )
                batch_mask[sentence_index, : len(sentence_mask)] = torch.tensor(
                    sentence_mask,
                    dtype=torch.bool,
                )

            batch_masks.append(batch_mask)

        return tuple(batch_masks)
