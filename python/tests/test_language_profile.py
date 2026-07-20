from pathlib import Path

from prism.data.treebanks import UniversalDependenciesTreebankSpec
from prism.languages import LanguageProfileSpec
from prism.modeling import PretrainedBackboneSpec


def test_language_profile_references_replaceable_student_backbone() -> None:
    backbone = PretrainedBackboneSpec(
        model_id="example/model",
        revision="a" * 40,
        trust_remote_code=False,
    )

    treebank = UniversalDependenciesTreebankSpec(
        repository_id="example/treebank",
        revision="a" * 40,
        license_id="CC-BY-SA-4.0",
        training_path=Path("data/train.conllu"),
        development_path=Path("data/dev.conllu"),
    )

    profile = LanguageProfileSpec(
        language_tag="nb",
        display_name="Norwegian Bokmål",
        student_backbone=backbone,
        gold_treebank=treebank,
    )

    assert profile.language_tag == "nb"
    assert profile.display_name == "Norwegian Bokmål"
    assert profile.student_backbone is backbone
    assert profile.gold_treebank is treebank
