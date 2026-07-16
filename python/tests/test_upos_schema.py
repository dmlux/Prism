import pytest

from prism.schema import (
    UPOS_SCHEMA_VERSION,
    build_upos_schema,
)


def test_build_upos_schema_is_deterministic() -> None:
    schema = build_upos_schema(
        [
            "VERB",
            "NOUN",
            "ADJ",
            "VERB",
        ]
    )

    assert schema.version == UPOS_SCHEMA_VERSION
    assert schema.labels == (
        "ADJ",
        "NOUN",
        "VERB",
    )
    assert schema.label_id_for("NOUN") == 1
    assert schema.label_for_id(1) == "NOUN"
    assert schema.label_id_for("PRON") is None

    with pytest.raises(ValueError, match="out of range"):
        schema.label_for_id(99)
