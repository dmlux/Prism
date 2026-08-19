"""Tests for the shared bordered-table progress logger (prism.progress)."""

from __future__ import annotations

from prism.progress import Column, ProgressLogger


def _logger() -> ProgressLogger:
    return ProgressLogger(
        columns=[
            Column("tokens", "tokens", kind="count", width=13),
            Column("loss", "loss"),
            Column("eval_loss", "eval_loss"),
            Column("perplexity", "perplexity"),
        ],
        row_kinds={"train": ["tokens", "loss"], "eval": ["tokens", "eval_loss", "perplexity"]},
    )


def test_header_and_row_are_bordered_and_aligned() -> None:
    log = _logger()
    counters = [("step", 2000, 100000)]
    header = log.format_header("train", counters=counters)
    row = log.format_row("train", counters=counters, values={"tokens": 131072000, "loss": 2.587})

    top, labels, sep = header.splitlines()
    assert top.startswith("┌") and top.endswith("┐")
    assert sep.startswith("├") and sep.endswith("┤")
    assert labels.startswith("│") and labels.endswith("│")
    assert row.startswith("│") and row.endswith("│")

    # Every line is the same visual width and the column separators sit at the
    # same positions -> values line up exactly under their header.
    assert len({len(top), len(labels), len(sep), len(row)}) == 1
    bar_positions = lambda s: [i for i, c in enumerate(s) if c == "│"]
    assert bar_positions(labels) == bar_positions(row)


def test_multi_kind_labels_and_values_render() -> None:
    log = _logger()
    labels = log.format_header("train", counters=[("step", 1, 10)]).splitlines()[1]
    assert "phase" in labels and "step" in labels and "tokens" in labels and "loss" in labels

    row = log.format_row(
        "eval",
        counters=[("step", 5, 10)],
        values={"tokens": 1234567, "eval_loss": 2.5, "perplexity": 12.182494},
    )
    assert "eval" in row
    assert "1,234,567" in row        # count -> thousands separator
    assert "2.500000" in row         # number -> fixed 6 decimals
    assert "12.182494" in row


def test_single_kind_has_no_phase_column() -> None:
    log = ProgressLogger(columns=[Column("acc", "acc")], row_kinds={"train": ["acc"]})
    assert "phase" not in log.format_header("train")
    assert log.format_row("train", values={"acc": 0.5}).count("│") == 2  # one column, two borders


def test_one_header_per_table_no_midtable_headers(capsys) -> None:
    """A block of same-kind rows gets exactly one header (at the top), never a
    mid-table repeat; close() frames it, and the next block opens a fresh one."""
    log = _logger()
    for step in range(1, 6):  # 5 train rows -> one table
        log.log("train", counters=[("step", step, 10)], values={"tokens": step * 100, "loss": 1.0 / step})
    log.close()
    for step in range(6, 9):  # a fresh 3-row train table
        log.log("train", counters=[("step", step, 10)], values={"tokens": step * 100, "loss": 1.0 / step})
    log.close()

    out = capsys.readouterr().out
    assert out.count("┌") == 2 and out.count("┐") == 2   # two tables opened
    assert out.count("└") == 2 and out.count("┘") == 2   # both closed
    header_lines = [l for l in out.splitlines() if "tokens" in l and "loss" in l]
    assert len(header_lines) == 2                          # exactly one header per table
    value_rows = [l for l in out.splitlines() if l.startswith("│") and "tokens" not in l]
    assert len(value_rows) == 8


def test_kind_switch_closes_previous_and_opens_new(capsys) -> None:
    log = _logger()
    log.log("train", counters=[("step", 1, 10)], values={"tokens": 100, "loss": 1.0})
    log.log("eval", counters=[("step", 1, 10)], values={"tokens": 100, "eval_loss": 0.5, "perplexity": 1.6})
    log.close()

    out = capsys.readouterr().out
    # train table opened+closed (on the switch), eval table opened+closed (on close).
    assert out.count("┌") == 2
    assert out.count("└") == 2


def test_close_writes_bottom_border_and_is_idempotent(capsys) -> None:
    log = _logger()
    log.log("train", counters=[("step", 1, 10)], values={"tokens": 100, "loss": 1.0})
    log.close()
    log.close()  # idempotent — no further output

    out = capsys.readouterr().out
    assert out.count("└") == 1
    assert out.rstrip().endswith("┘")  # the very last line is the bottom border


def test_unknown_kind_raises() -> None:
    log = _logger()
    try:
        log.format_row("nope", values={})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown row kind")
