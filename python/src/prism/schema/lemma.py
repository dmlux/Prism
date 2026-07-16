from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from types import MappingProxyType


LEMMA_RULE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class LemmaEditRule:
    prefix_removal: int
    suffix_removal: int
    prefix_addition: str
    suffix_addition: str

    def __post_init__(self) -> None:
        if self.prefix_removal < 0:
            raise ValueError("Lemma prefix removal must not be negative.")
        if self.suffix_removal < 0:
            raise ValueError("Lemma suffix removal must not be negative.")

    def apply(self, token: str) -> str:
        if self.prefix_removal + self.suffix_removal > len(token):
            raise ValueError(
                "Lemma edit rule removes more characters than the token contains."
            )

        unchanged_end = len(token) - self.suffix_removal
        unchanged_text = token[self.prefix_removal : unchanged_end]

        return self.prefix_addition + unchanged_text + self.suffix_addition

    @property
    def sort_key(self) -> tuple[int, int, str, str]:
        return (
            self.prefix_removal,
            self.suffix_removal,
            self.prefix_addition,
            self.suffix_addition,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LemmaRuleSchema:
    version: int
    rules: tuple[LemmaEditRule, ...]

    _rule_ids: Mapping[LemmaEditRule, int] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Lemma rule schema version must be positive.")
        if not self.rules:
            raise ValueError("Lemma rule schema must contain rules.")
        if len(set(self.rules)) != len(self.rules):
            raise ValueError("Lemma rules must be unique.")
        if self.rules != tuple(
            sorted(
                self.rules,
                key=lambda rule: rule.sort_key,
            )
        ):
            raise ValueError("Lemma rules must be sorted.")
        object.__setattr__(
            self,
            "_rule_ids",
            MappingProxyType(
                {rule: rule_id for rule_id, rule in enumerate(self.rules)}
            ),
        )

    def rule_id_for(
        self,
        rule: LemmaEditRule,
    ) -> int | None:
        return self._rule_ids.get(rule)

    def rule_for_id(
        self,
        rule_id: int,
    ) -> LemmaEditRule:
        if rule_id < 0 or rule_id >= len(self.rules):
            raise ValueError(f"Lemma rule ID {rule_id} is out of range.")

        return self.rules[rule_id]


def derive_lemma_edit_rule(
    token: str,
    lemma: str,
) -> LemmaEditRule:
    match = SequenceMatcher(
        a=token,
        b=lemma,
        autojunk=False,
    ).find_longest_match()

    return LemmaEditRule(
        prefix_removal=match.a,
        suffix_removal=(len(token) - match.a - match.size),
        prefix_addition=lemma[: match.b],
        suffix_addition=lemma[match.b + match.size :],
    )
