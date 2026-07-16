import pytest

from prism.schema import (
    LEMMA_RULE_SCHEMA_VERSION,
    LemmaEditRule,
    build_lemma_rule_schema,
    derive_lemma_edit_rule,
)


def test_build_lemma_rule_schema_is_deterministic() -> None:
    token_lemma_pairs = [
        ("hus", "hus"),
        ("bok", "bok"),
        ("husene", "hus"),
        ("gikk", "gå"),
    ]

    schema = build_lemma_rule_schema(token_lemma_pairs)

    assert schema.version == LEMMA_RULE_SCHEMA_VERSION
    assert schema.rules == (
        LemmaEditRule(
            prefix_removal=0,
            suffix_removal=0,
            prefix_addition="",
            suffix_addition="",
        ),
        LemmaEditRule(
            prefix_removal=0,
            suffix_removal=3,
            prefix_addition="",
            suffix_addition="",
        ),
        LemmaEditRule(
            prefix_removal=0,
            suffix_removal=3,
            prefix_addition="",
            suffix_addition="å",
        ),
    )


def test_lemma_rule_schema_maps_rules_and_class_ids() -> None:
    schema = build_lemma_rule_schema(
        [
            ("hus", "hus"),
            ("husene", "hus"),
        ]
    )
    identity_rule = derive_lemma_edit_rule("hus", "hus")
    unknown_rule = LemmaEditRule(
        prefix_removal=1,
        suffix_removal=0,
        prefix_addition="",
        suffix_addition="",
    )

    assert schema.rule_id_for(identity_rule) == 0
    assert schema.rule_for_id(0) == identity_rule
    assert schema.rule_id_for(unknown_rule) is None

    with pytest.raises(ValueError, match="out of range"):
        schema.rule_for_id(99)
