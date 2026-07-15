from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True, slots=True, kw_only=True)
class LemmaEditRule:
    prefix_removal: int
    suffix_removal: int
    prefix_addition: str
    suffix_addition: str

    def __post_init__(self) -> None:
        if self.prefix_removal < 0:
            raise ValueError(
                "Lemma prefix removal must not be negative."
            )
        if self.suffix_removal < 0:
            raise ValueError(
                "Lemma suffix removal must not be negative."
            )

    def apply(self, token: str) -> str:
        if (
            self.prefix_removal + self.suffix_removal
            > len(token)
        ):
            raise ValueError(
                "Lemma edit rule removes more characters "
                "than the token contains."
            )

        unchanged_end = len(token) - self.suffix_removal
        unchanged_text = token[
            self.prefix_removal:unchanged_end
        ]

        return (
            self.prefix_addition
            + unchanged_text
            + self.suffix_addition
        )



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
        suffix_removal=(
            len(token) - match.a - match.size
        ),
        prefix_addition=lemma[:match.b],
        suffix_addition=lemma[
            match.b + match.size:
        ],
    )