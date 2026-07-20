from prism.languages.norwegian import (
    NORBERT4_BASE_BACKBONE,
    NORBERT4_XSMALL_BACKBONE,
    NORWEGIAN_BOKMAAL_PROFILE,
    NORWEGIAN_NYNORSK_PROFILE,
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


def test_norbert4_base_teacher_backbone_is_reproducibly_pinned() -> None:
    assert NORBERT4_BASE_BACKBONE.model_id == "ltg/norbert4-base"
    assert NORBERT4_BASE_BACKBONE.revision == (
        "386ba2dc5ae5f95fec86d580c5fc4af34d380126"
    )
    assert NORBERT4_BASE_BACKBONE.trust_remote_code is True


def test_norwegian_profiles_select_shared_norbert4_teacher() -> None:
    assert NORWEGIAN_BOKMAAL_PROFILE.teacher_backbone is NORBERT4_BASE_BACKBONE
    assert NORWEGIAN_NYNORSK_PROFILE.teacher_backbone is NORBERT4_BASE_BACKBONE


def test_language_profile_selects_backbone_by_model_role() -> None:
    assert (
        NORWEGIAN_BOKMAAL_PROFILE.backbone_for_role("student")
        is NORBERT4_XSMALL_BACKBONE
    )
    assert (
        NORWEGIAN_BOKMAAL_PROFILE.backbone_for_role("teacher") is NORBERT4_BASE_BACKBONE
    )
