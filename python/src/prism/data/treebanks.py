"""Typed dataset contracts for Universal Dependencies treebanks."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class UniversalDependenciesTreebankSpec:
    repository_id: str
    revision: str
    license_id: str
    training_path: Path
    development_path: Path
