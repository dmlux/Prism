"""Central training-progress logger — one consistent, bordered-table format for
every training loop in the repo.

Both the PrismBERT backbone pretraining (:mod:`prism.bert.pretrain`) and the
per-language tagger training (:mod:`prism.languages`) feed structured values to a
:class:`ProgressLogger` instead of hand-formatting their own lines, so the
on-disk logs never drift apart in style. Each component declares its own columns
(the metrics differ); the *style* — a box-drawn table with a repeated header row,
values right-aligned directly under their column header, fixed-decimal floats and
thousands-separated counts — lives here, once.

Rows are grouped by *kind* (e.g. ``"train"`` / ``"eval"``); each kind renders its
own table. A header block (top border, column headers, separator) is re-emitted
whenever the kind changes or every ``header_every`` rows, so a value row in a long
scrolling log always has its column headers within view — you never have to guess
which value belongs to which label.

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
    # ┌───────┬─────────────┬───────────────┬──────────┐
    # │ phase │        step │        tokens │     loss │
    # ├───────┼─────────────┼───────────────┼──────────┤
    # │ train │ 2000/100000 │   131,072,000 │ 2.587000 │
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_NUMBER = "number"
_COUNT = "count"
_TEXT = "text"

# Box-drawing pieces.
_H = "─"
_V = "│"
_TOP_L, _TOP_M, _TOP_R = "┌", "┬", "┐"
_MID_L, _MID_M, _MID_R = "├", "┼", "┤"
_BOT_L, _BOT_M, _BOT_R = "└", "┴", "┘"


@dataclass(frozen=True)
class Column:
    """One column in the progress table.

    ``kind`` selects the value formatting: ``"number"`` → blank-padded float with
    ``decimals`` places; ``"count"`` → thousands-separated integer; ``"text"`` →
    right-justified string. ``width`` is the value field width (blanks, never
    leading zeros); the actual column is widened to fit its header label too.
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
    """Formats and prints bordered progress tables in one shared style.

    Configure once with the full column set and, per *row kind* (e.g. ``"train"``
    and ``"eval"``), the ordered column keys that kind uses. Each ``log`` call
    prints a value row; a header block is (re-)printed whenever the kind changes
    or every ``header_every`` rows so the headers stay in view. When more than one
    kind is configured, a leading ``phase`` column names the kind on each row.
    """

    def __init__(
        self,
        columns: Sequence[Column],
        row_kinds: Mapping[str, Sequence[str]],
        *,
        header_every: int = 20,
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
        self._header_every = max(1, header_every)
        self._multi_kind = len(self._row_kinds) > 1
        self._phase_width = max((len(kind) for kind in self._row_kinds), default=0)
        self._counts: dict[str, int] = {kind: 0 for kind in self._row_kinds}
        self._last_kind: str | None = None
        # Column widths of the table section currently left open (awaiting its
        # bottom border); None when no section is open.
        self._open_widths: list[int] | None = None

    def _cells(
        self,
        kind: str,
        tag: str | None,
        counters: Sequence[tuple[str, int, int]],
        values: Mapping[str, object],
    ) -> tuple[list[str], list[str], list[int]]:
        """Return parallel (headers, rendered values, column widths). Widths are
        value-independent (max of header and the column's field width), so header
        and value rows always align and rows between two headers stay in grid."""
        headers: list[str] = []
        cells: list[str] = []
        widths: list[int] = []
        if self._multi_kind:
            headers.append("phase")
            cells.append(tag if tag is not None else kind)
            widths.append(max(self._phase_width, len("phase")))
        for label, current, total in counters:
            total_str = str(total)
            headers.append(label)
            cells.append(f"{current:>{len(total_str)}}/{total_str}")
            widths.append(max(len(label), 2 * len(total_str) + 1))
        for key in self._row_kinds[kind]:
            column = self._columns[key]
            headers.append(column.label)
            cells.append(column.render_value(values[key]) if key in values else "")
            widths.append(max(len(column.label), column.width))
        return headers, cells, widths

    @staticmethod
    def _rule(widths: Sequence[int], left: str, mid: str, right: str) -> str:
        return left + mid.join(_H * (width + 2) for width in widths) + right

    @staticmethod
    def _line(items: Sequence[str], widths: Sequence[int]) -> str:
        return _V + _V.join(f" {item:>{width}} " for item, width in zip(items, widths)) + _V

    def format_header(
        self,
        kind: str,
        *,
        tag: str | None = None,
        counters: Sequence[tuple[str, int, int]] = (),
        values: Mapping[str, object] | None = None,
    ) -> str:
        """The three-line header block (top border, column labels, separator)."""
        if kind not in self._row_kinds:
            raise ValueError(f"Unknown row kind {kind!r}.")
        headers, _, widths = self._cells(kind, tag, counters, values or {})
        return "\n".join(
            (
                self._rule(widths, _TOP_L, _TOP_M, _TOP_R),
                self._line(headers, widths),
                self._rule(widths, _MID_L, _MID_M, _MID_R),
            )
        )

    def format_row(
        self,
        kind: str,
        *,
        tag: str | None = None,
        counters: Sequence[tuple[str, int, int]] = (),
        values: Mapping[str, object],
    ) -> str:
        """A single bordered value row (values right-aligned under their header)."""
        if kind not in self._row_kinds:
            raise ValueError(f"Unknown row kind {kind!r}.")
        _, cells, widths = self._cells(kind, tag, counters, values)
        return self._line(cells, widths)

    def log(
        self,
        kind: str,
        *,
        tag: str | None = None,
        counters: Sequence[tuple[str, int, int]] = (),
        values: Mapping[str, object],
    ) -> None:
        if kind not in self._row_kinds:
            raise ValueError(f"Unknown row kind {kind!r}.")
        headers, cells, widths = self._cells(kind, tag, counters, values)
        if kind != self._last_kind:
            # Start a new table section: close the previous kind's table with a
            # bottom border (if one is open), then open this one.
            if self._open_widths is not None:
                print(self._rule(self._open_widths, _BOT_L, _BOT_M, _BOT_R), flush=True)
            print(self._rule(widths, _TOP_L, _TOP_M, _TOP_R), flush=True)
            print(self._line(headers, widths), flush=True)
            print(self._rule(widths, _MID_L, _MID_M, _MID_R), flush=True)
            self._open_widths = widths
        elif self._counts[kind] % self._header_every == 0:
            # Repeat the header inside the open section (a single mid rule, no
            # bottom border — the section stays open).
            print(self._rule(widths, _MID_L, _MID_M, _MID_R), flush=True)
            print(self._line(headers, widths), flush=True)
            print(self._rule(widths, _MID_L, _MID_M, _MID_R), flush=True)
        print(self._line(cells, widths), flush=True)
        self._counts[kind] += 1
        self._last_kind = kind

    def close(self) -> None:
        """Write the bottom border of the still-open table section — call once
        when the training loop ends so the last table is framed. Idempotent; the
        logger may be reused afterwards (the next :meth:`log` opens a fresh
        section)."""
        if self._open_widths is not None:
            print(self._rule(self._open_widths, _BOT_L, _BOT_M, _BOT_R), flush=True)
            self._open_widths = None
            self._last_kind = None
