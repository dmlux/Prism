from pathlib import Path

import pytest
import torch

from prism.data import PretokenizedSentence
from prism.data.silver import PretokenizedSilverSentence
from prism.languages.norwegian.label_silver_corpus import (
    _budgeted_sentences,
    parse_labeling_arguments,
)
from prism.training import (
    SILVER_LABEL_FORMAT_VERSION,
    SilverLabelManifest,
    SilverLabelShardReference,
    SilverSentenceLabels,
    iter_silver_label_records,
    load_silver_label_manifest,
    write_silver_label_manifest,
    write_silver_label_shard,
)


def _labels(
    document_id: str,
    sentence_index: int,
    token_count: int,
    *,
    with_agreement: bool = True,
) -> SilverSentenceLabels:
    return SilverSentenceLabels(
        document_id=document_id,
        sentence_index=sentence_index,
        upos_probabilities=torch.rand((token_count, 17), dtype=torch.float16),
        morphology_probabilities=(
            torch.rand((token_count, 3), dtype=torch.float16),
            torch.rand((token_count, 4), dtype=torch.float16),
        ),
        lemma_rule_ids=torch.arange(token_count * 8, dtype=torch.int32).reshape(
            token_count, 8
        ),
        lemma_rule_probabilities=torch.rand(
            (token_count, 8),
            dtype=torch.float16,
        ),
        agreement_upos_ids=(
            torch.zeros(token_count, dtype=torch.int16) if with_agreement else None
        ),
        agreement_lemma_rule_ids=(
            torch.zeros(token_count, dtype=torch.int32) if with_agreement else None
        ),
        agreement_morphology_predictions=(
            (
                torch.zeros((token_count, 4), dtype=torch.bool),
                torch.zeros((token_count, 5), dtype=torch.bool),
            )
            if with_agreement
            else None
        ),
    )


def test_silver_label_shard_round_trip(tmp_path: Path) -> None:
    records = (_labels("urn:1", 0, 5), _labels("urn:1", 1, 3))
    shard_path = tmp_path / "labels-00001.pt"

    reference = write_silver_label_shard(records=records, path=shard_path)
    manifest = SilverLabelManifest(
        format_version=SILVER_LABEL_FORMAT_VERSION,
        corpus_path="data/processed/sakspapir-nno/pretokenized.jsonl",
        corpus_sha256="0" * 64,
        corpus_manifest={"corpus_id": "oai:nb.no:sbr-60"},
        labeler_checkpoint_path="runs/teacher/best.pt",
        labeler_checkpoint_sha256="1" * 64,
        labeler_epoch_index=11,
        calibration={"upos_temperature": 2.48},
        morphology_logit_correction_strength=1.0,
        agreement_checkpoint_path=None,
        agreement_checkpoint_sha256=None,
        agreement_logit_correction_strength=None,
        lemma_top_k=8,
        probability_dtype="float16",
        maximum_token_budget=1000,
        sentence_count=2,
        token_count=8,
        shards=(reference,),
    )

    loaded = tuple(
        iter_silver_label_records(
            directory=tmp_path,
            manifest=manifest,
            verify_checksums=True,
        )
    )

    assert reference.sentence_count == 2
    assert reference.token_count == 8
    assert len(loaded) == 2
    assert loaded[0].document_id == "urn:1"
    assert loaded[1].token_count == 3
    torch.testing.assert_close(
        loaded[0].upos_probabilities,
        records[0].upos_probabilities,
    )
    assert loaded[0].agreement_upos_ids is not None


def test_silver_label_manifest_round_trip(tmp_path: Path) -> None:
    reference = SilverLabelShardReference(
        file_name="labels-00001.pt",
        sentence_count=2,
        token_count=8,
        sha256="a" * 64,
    )
    manifest = SilverLabelManifest(
        format_version=SILVER_LABEL_FORMAT_VERSION,
        corpus_path="data/processed/sakspapir-nno/pretokenized.jsonl",
        corpus_sha256="0" * 64,
        corpus_manifest={"corpus_id": "oai:nb.no:sbr-60"},
        labeler_checkpoint_path="runs/teacher/best.pt",
        labeler_checkpoint_sha256="1" * 64,
        labeler_epoch_index=11,
        calibration={"upos_temperature": 2.48},
        morphology_logit_correction_strength=1.0,
        agreement_checkpoint_path="runs/control/best.pt",
        agreement_checkpoint_sha256="2" * 64,
        agreement_logit_correction_strength=1.0,
        lemma_top_k=8,
        probability_dtype="float16",
        maximum_token_budget=None,
        sentence_count=2,
        token_count=8,
        shards=(reference,),
    )
    path = tmp_path / "labels-manifest.json"

    write_silver_label_manifest(manifest, path)
    loaded = load_silver_label_manifest(path)

    assert loaded == manifest


def test_silver_label_manifest_rejects_inconsistent_counts() -> None:
    reference = SilverLabelShardReference(
        file_name="labels-00001.pt",
        sentence_count=2,
        token_count=8,
        sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="sentence count"):
        SilverLabelManifest(
            format_version=SILVER_LABEL_FORMAT_VERSION,
            corpus_path="corpus.jsonl",
            corpus_sha256="0" * 64,
            corpus_manifest={},
            labeler_checkpoint_path="runs/teacher/best.pt",
            labeler_checkpoint_sha256="1" * 64,
            labeler_epoch_index=0,
            calibration={},
            morphology_logit_correction_strength=1.0,
            agreement_checkpoint_path=None,
            agreement_checkpoint_sha256=None,
            agreement_logit_correction_strength=None,
            lemma_top_k=8,
            probability_dtype="float16",
            maximum_token_budget=None,
            sentence_count=3,
            token_count=8,
            shards=(reference,),
        )


def test_sentence_labels_reject_partial_agreement() -> None:
    with pytest.raises(ValueError, match="complete or absent"):
        labels = _labels("urn:1", 0, 2)
        SilverSentenceLabels(
            document_id=labels.document_id,
            sentence_index=labels.sentence_index,
            upos_probabilities=labels.upos_probabilities,
            morphology_probabilities=labels.morphology_probabilities,
            lemma_rule_ids=labels.lemma_rule_ids,
            lemma_rule_probabilities=labels.lemma_rule_probabilities,
            agreement_upos_ids=labels.agreement_upos_ids,
            agreement_lemma_rule_ids=None,
            agreement_morphology_predictions=None,
        )


def _silver_sentence(index: int, token_count: int) -> PretokenizedSilverSentence:
    tokens = tuple(f"tok{position}" for position in range(token_count))
    return PretokenizedSilverSentence(
        document_id="urn:1",
        sentence_index=index,
        model_input=PretokenizedSentence(
            tokens=tokens,
            has_space_before=(False,) + (True,) * (token_count - 1),
        ),
    )


def test_budgeted_sentences_include_the_crossing_sentence() -> None:
    sentences = tuple(_silver_sentence(index, 4) for index in range(5))

    unlimited = tuple(_budgeted_sentences(iter(sentences), None))
    budgeted = tuple(_budgeted_sentences(iter(sentences), 9))

    assert len(unlimited) == 5
    # 4 + 4 tokens reach 8 < 9, the third sentence crosses the budget and is
    # included; the fourth is not.
    assert len(budgeted) == 3


def test_parse_labeling_arguments_validates_agreement_pairing() -> None:
    base = (
        "--silver-corpus",
        "data/processed/sakspapir-nno/pretokenized.jsonl",
        "--silver-manifest",
        "data/processed/sakspapir-nno/manifest.json",
        "--checkpoint",
        "runs/teacher/best.pt",
        "--calibration",
        "runs/teacher/calibration-corrected.json",
        "--output-directory",
        "data/processed/sakspapir-nno/labels",
    )

    arguments = parse_labeling_arguments(
        (
            *base,
            "--morphology-logit-correction-strength",
            "1.0",
            "--maximum-token-budget",
            "1000000",
        )
    )
    assert arguments.maximum_token_budget == 1000000
    assert arguments.lemma_top_k == 8
    assert arguments.agreement_checkpoint_path is None

    with pytest.raises(SystemExit):
        parse_labeling_arguments(
            (*base, "--agreement-checkpoint", "runs/control/best.pt")
        )
