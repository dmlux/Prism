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
    alternate_teacher_backbones: tuple[PretrainedBackboneSpec, ...] = ()

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
        known_model_ids = [
            backbone.model_id for backbone in self.backbones_for_role("teacher")
        ]
        if len(set(known_model_ids)) != len(known_model_ids):
            raise ValueError("Teacher backbone model IDs must be unique.")

    def backbone_for_role(
        self,
        role: ModelRole,
    ) -> PretrainedBackboneSpec:
        return self.backbones_for_role(role)[0]

    def backbones_for_role(
        self,
        role: ModelRole,
    ) -> tuple[PretrainedBackboneSpec, ...]:
        """Return the role's default backbone plus any alternate variants."""

        if role == "student":
            return (self.student_backbone,)
        if role == "teacher":
            return (self.teacher_backbone, *self.alternate_teacher_backbones)

        raise ValueError(f"Unsupported model role: {role}")

    def backbone_for_model_id(
        self,
        model_id: str,
        *,
        role: ModelRole,
    ) -> PretrainedBackboneSpec:
        """Resolve a checkpoint's stored backbone model ID to a pinned spec.

        Checkpoints record their backbone identity; loading resolves it among
        the profile's known backbones for the role instead of assuming the
        role default, so alternate teacher variants stay loadable.
        """

        for backbone in self.backbones_for_role(role):
            if backbone.model_id == model_id:
                return backbone
        raise ValueError(
            f"Backbone {model_id!r} is not a known {role} backbone of the "
            f"{self.display_name} profile."
        )
