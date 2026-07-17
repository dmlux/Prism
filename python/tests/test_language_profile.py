from prism.languages import LanguageProfileSpec
from prism.modeling import PretrainedBackboneSpec


def test_language_profile_references_replaceable_student_backbone() -> None:
    backbone = PretrainedBackboneSpec(
        model_id="example/model",
        revision="a" * 40,
        trust_remote_code=False,
    )

    profile = LanguageProfileSpec(
        language_tag="nb",
        display_name="Norwegian Bokmål",
        student_backbone=backbone,
    )

    assert profile.language_tag == "nb"
    assert profile.display_name == "Norwegian Bokmål"
    assert profile.student_backbone is backbone
