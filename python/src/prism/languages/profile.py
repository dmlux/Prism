from dataclasses import dataclass

from prism.data.treebanks import UniversalDependenciesTreebankSpec
from prism.modeling.backbones import PretrainedBackboneSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class LanguageProfileSpec:
    language_tag: str
    display_name: str
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
