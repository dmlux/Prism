"""Narrow observer contract for token-task evaluation predictions."""

from typing import Protocol

from prism.modeling.outputs import TokenTaskPredictionBatch


class TokenTaskPredictionObserver(Protocol):
    def add(self, *, predictions: TokenTaskPredictionBatch) -> None: ...
