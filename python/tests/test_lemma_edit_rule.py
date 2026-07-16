import pytest

from prism.schema import (
    LemmaEditRule,
    derive_lemma_edit_rule,
)


def test_lemma_edit_rule_applies_suffix_edits() -> None:
    plural_rule = LemmaEditRule(
        prefix_removal=0,
        suffix_removal=3,
        prefix_addition="",
        suffix_addition="",
    )
    irregular_rule = LemmaEditRule(
        prefix_removal=0,
        suffix_removal=3,
        prefix_addition="",
        suffix_addition="å",
    )

    assert plural_rule.apply("husene") == "hus"
    assert irregular_rule.apply("gikk") == "gå"


def test_lemma_edit_rule_rejects_invalid_removals() -> None:
    with pytest.raises(ValueError, match="prefix removal"):
        LemmaEditRule(
            prefix_removal=-1,
            suffix_removal=0,
            prefix_addition="",
            suffix_addition="",
        )

    with pytest.raises(ValueError, match="suffix removal"):
        LemmaEditRule(
            prefix_removal=0,
            suffix_removal=-1,
            prefix_addition="",
            suffix_addition="",
        )

    rule = LemmaEditRule(
        prefix_removal=2,
        suffix_removal=2,
        prefix_addition="",
        suffix_addition="",
    )

    with pytest.raises(ValueError, match="more characters"):
        rule.apply("hus")


@pytest.mark.parametrize(
    ("token", "lemma", "expected_rule"),
    [
        (
            "husene",
            "hus",
            LemmaEditRule(
                prefix_removal=0,
                suffix_removal=3,
                prefix_addition="",
                suffix_addition="",
            ),
        ),
        (
            "gikk",
            "gå",
            LemmaEditRule(
                prefix_removal=0,
                suffix_removal=3,
                prefix_addition="",
                suffix_addition="å",
            ),
        ),
        (
            "forhånd",
            "hånd",
            LemmaEditRule(
                prefix_removal=3,
                suffix_removal=0,
                prefix_addition="",
                suffix_addition="",
            ),
        ),
    ],
)
def test_derive_lemma_edit_rule_preserves_longest_shared_text(
    token: str,
    lemma: str,
    expected_rule: LemmaEditRule,
) -> None:
    rule = derive_lemma_edit_rule(token, lemma)

    assert rule == expected_rule
    assert rule.apply(token) == lemma
