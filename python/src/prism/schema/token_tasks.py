from dataclasses import dataclass

from prism.schema.lemma import LemmaRuleSchema
from prism.schema.morphology import MorphologySchema
from prism.schema.upos import UposSchema


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenTaskSchema:
    upos: UposSchema
    morphology: MorphologySchema
    lemma_rules: LemmaRuleSchema
