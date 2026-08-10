"""Central training-progress logger — one consistent, aligned, table-like line
format for every training loop in the repo.

Both the PrismBERT backbone pretraining (:mod:`prism.bert.pretrain`) and the
per-language tagger training (:mod:`prism.languages`) feed structured values to
a :class:`ProgressLogger` instead of hand-formatting their own lines, so the
on-disk logs never drift apart in style. Each component declares its own columns
(the metrics differ), but the *style* — spelled-out labels, blank-padded numbers
with fixed decimals, thousands-separated counts, and columns that line up across
row kinds — lives here, once.

Dependency-light on purpose (standard library only, no torch), so any module can
import it cheaply.

Example::

    logger = ProgressLogger(
        columns=[
            Column("tokens", "tokens", kind="count", width=13),
            Column("loss", "loss"),
            Column("eval_loss", "eval_loss"),
            Column("perplexity", "perplexity"),
        ],
        row_kinds={"train": ["tokens", "loss"], "eval": ["tokens", "eval_loss", "perplexity"]},
    )
    logger.log("train", counters=[("step", 2000, 100000)], values={"tokens": 131072000, "loss": 2.587})
    # [train] step   2000/100000    tokens   131,072,000    loss      2.587000
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_NUMBER = "number"
_COUNT = "count"
_TEXT = "text"


@dataclass(frozen=True)
class Column:
    """One column in the progress table.

    ``kind`` selects the value formatting: ``"number"`` → blank-padded float with
    ``decimals`` places; ``"count"`` → thousands-separated integer; ``"text"`` →
    right-justified string. ``width`` is the value field width (blanks, never
    leading zeros); the label is padded separately to a per-position width that
    the logger computes so that different row kinds line up.
    """

    key: str
    label: str
    kind: str = _NUMBER
    width: int = 13
    decimals: int = 6

    def render_value(self, value: object) -> str:
        if self.kind == _COUNT:
            return f"{int(value):,}".rjust(self.width)
        if self.kind == _TEXT:
            return f"{str(value):>{self.width}}"
        return f"{float(value):>{self.width}.{self.decimals}f}"


class ProgressLogger:
    """Formats and prints aligned, table-like progress rows in one shared style.

    Configure once with the full column set and, per *row kind* (e.g. ``"train"``
    and ``"eval"``), the ordered column keys that kind uses. The label width of
    each column *position* is the widest label any kind places there, so rows of
    different kinds line up as a single grid (e.g. ``loss`` and ``eval_loss``
    occupy the same-width column). Then just feed data with :meth:`log`.
    """

    def __init__(
        self,
        columns: Sequence[Column],
        row_kinds: Mapping[str, Sequence[str]],
        *,
        separator: str = "    ",
    ) -> None:
        self._columns = {column.key: column for column in columns}
        if len(self._columns) != len(columns):
            raise ValueError("Column keys must be unique.")
        self._row_kinds = {kind: tuple(keys) for kind, keys in row_kinds.items()}
        for kind, keys in self._row_kinds.items():
            missing = [key for key in keys if key not in self._columns]
            if missing:
                raise ValueError(f"Row kind {kind!r} references unknown columns {missing}.")
        if not self._row_kinds:
            raise ValueError("At least one row kind is required.")
        self._separator = separator
        self._tag_width = max(len(kind) for kind in self._row_kinds)
        # Per-position label width = widest label any row kind uses at that
        # position, so columns line up across kinds.
        position_count = max(len(keys) for keys in self._row_kinds.values())
        self._position_label_width = [
            max(
                (len(self._columns[keys[position]].label) for keys in self._row_kinds.values()
                 if position < len(keys)),
                default=0,
            )
            for position in range(position_count)
        ]

    def _counter(self, label: str, current: int, total: int) -> str:
        return f"{label} {current:>{len(str(total))}d}/{total}"

    def format_row(
        self,
        kind: str,
        *,
        tag: str | None = None,
        counters: Sequence[tuple[str, int, int]] = (),
        values: Mapping[str, object],
    ) -> str:
        if kind not in self._row_kinds:
            raise ValueError(f"Unknown row kind {kind!r}.")
        parts = [self._counter(label, current, total) for label, current, total in counters]
        for position, key in enumerate(self._row_kinds[kind]):
            column = self._columns[key]
            label_width = self._position_label_width[position]
            if key in values:
                parts.append(f"{column.label:<{label_width}} {column.render_value(values[key])}")
            else:
                # Keep the grid: blank cell of the same width as a rendered one.
                parts.append(" " * (label_width + 1 + column.width))
        printed_tag = (tag if tag is not None else kind).ljust(self._tag_width)
        return f"[{printed_tag}] " + self._separator.join(parts)

    def log(
        self,
        kind: str,
        *,
        tag: str | None = None,
        counters: Sequence[tuple[str, int, int]] = (),
        values: Mapping[str, object],
    ) -> None:
        print(self.format_row(kind, tag=tag, counters=counters, values=values), flush=True)
