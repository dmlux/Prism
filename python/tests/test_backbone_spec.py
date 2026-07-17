from prism.languages.norwegian import (
    NORBERT4_XSMALL_BACKBONE,
    NORWEGIAN_BOKMAAL_PROFILE,
)


def test_norbert4_xsmall_backbone_is_reproducibly_pinned() -> None:
    assert NORBERT4_XSMALL_BACKBONE.model_id == "ltg/norbert4-xsmall"
    assert (
        NORBERT4_XSMALL_BACKBONE.revision == "7483327d36a2daa5dbe936c68aa277149c6f9632"
    )
    assert NORBERT4_XSMALL_BACKBONE.trust_remote_code is True


def test_norwegian_bokmaal_profile_selects_norbert4_student() -> None:
    assert NORWEGIAN_BOKMAAL_PROFILE.language_tag == "nb"
    assert NORWEGIAN_BOKMAAL_PROFILE.display_name == "Norwegian Bokmål"
    assert NORWEGIAN_BOKMAAL_PROFILE.student_backbone is NORBERT4_XSMALL_BACKBONE
