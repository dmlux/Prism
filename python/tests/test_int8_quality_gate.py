"""Unit tests for the shared int8 quality-gate engine.

The heavy end-to-end accuracy proof (reproducing the published prism-no
development deltas) needs a checkpoint and runs manually; these cover the
pure logic that turns decoded predictions into scored UD tokens.
"""

from prism.exporting.int8_quality_gate import (
    Int8QualityReport,
    TaskQualityDelta,
    _system_token,
)


def _decoded(form, upos, lemma, morphology):
    return {"form": form, "upos": upos, "lemma": lemma, "morphology": morphology}


def test_system_token_bundles_active_features_and_joins_multi_values() -> None:
    token = _system_token(
        _decoded(
            "bøker",
            "NOUN",
            "bok",
            {
                "Number": ["Plur"],
                "Gender": ["Fem", "Masc"],  # multi-valued
                "Case": [],  # inactive: dropped
            },
        ),
        lemma_decoder=lambda form, lemma, upos: lemma,
    )
    assert token.text == "bøker"
    assert token.upos == "NOUN"
    assert token.lemma == "bok"
    # Only active features, multi-values joined UD-style (sorted, comma).
    assert token.features == {"Number": "Plur", "Gender": "Fem,Masc"}


def test_system_token_applies_the_lemma_decoder() -> None:
    # A Norwegian-style marker-restoring decoder turns "," into "$,".
    def restore(form, lemma, upos):
        return "$" + lemma if form == "," else lemma

    token = _system_token(
        _decoded(",", "PUNCT", ",", {}),
        lemma_decoder=restore,
    )
    assert token.lemma == "$,"
    assert token.features == {}


def test_report_worst_regression_and_delta() -> None:
    report = Int8QualityReport(
        language_tag="xx",
        token_count=1000,
        tasks=(
            TaskQualityDelta(task="upos", fp32_accuracy=99.0, int8_accuracy=98.99, decision_flips=1),
            TaskQualityDelta(task="feats", fp32_accuracy=97.0, int8_accuracy=96.5, decision_flips=5),
            TaskQualityDelta(task="lemma", fp32_accuracy=99.0, int8_accuracy=99.1, decision_flips=2),
        ),
    )
    assert abs(report.tasks[0].delta - (-0.01)) < 1e-9
    # Worst (most negative) regression is the feats task.
    assert abs(report.worst_regression - (-0.5)) < 1e-9
    assert "xx development (1000 tokens)" in report.format_table()


def test_report_worst_regression_is_zero_when_int8_never_regresses() -> None:
    report = Int8QualityReport(
        language_tag="xx",
        token_count=10,
        tasks=(
            TaskQualityDelta(task="upos", fp32_accuracy=98.0, int8_accuracy=98.1, decision_flips=0),
        ),
    )
    assert report.worst_regression == 0.0
