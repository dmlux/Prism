from dataclasses import dataclass
from typing import Literal

from prism.data.treebanks import UniversalDependenciesTreebankSpec
from prism.modeling.backbones import PretrainedBackboneSpec


ModelRole = Literal["student", "teacher"]


@dataclass(frozen=True, slots=True, kw_only=True)
class LanguageProfileSpec:
    language_tag: str
    display_name: str
    teacher_backbone: PretrainedBackboneSpec
    student_backbone: PretrainedBackboneSpec
    gold_treebank: UniversalDependenciesTreebankSpec

    def __post_init__(self) -> None:
        if not self.language_tag or self.language_tag.strip() != self.language_tag:
            raise ValueError(
                "Language tag must be non-empty and have no surrounding whitespace."
            )
        if not self.display_name or self.display_name.strip() != self.display_name:
            raise ValueError(
                "Language display name must be non-empty "
                "and have no surrounding whitespace."
            )

    def backbone_for_role(
        self,
        role: ModelRole,
    ) -> PretrainedBackboneSpec:
        if role == "student":
            return self.student_backbone
        if role == "teacher":
            return self.teacher_backbone

        raise ValueError(f"Unsupported model role: {role}")
