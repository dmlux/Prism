import json
from pathlib import Path

import pytest
import torch
from torch import nn

from prism.modeling import TokenizedBatch, TokenTaskLogits
from prism.schema import MorphologyFeatureSchema, MorphologySchema
from prism.training import (
    SILVER_LABEL_FORMAT_VERSION,
    SilverFilterPolicy,
    SilverLabelManifest,
    SilverSentenceLabels,
    compute_silver_kd_loss,
    load_silver_training_sentences,
    train_mixed_token_task_epoch,
    write_silver_label_manifest,
    write_silver_label_shard,
)
from prism.languages.norwegian.train_baseline import parse_training_arguments


MORPHOLOGY_SCHEMA = MorphologySchema(
    version=1,
    features=(
        MorphologyFeatureSchema(
            name="Exclusive",
            values=("Value",),
            allows_multiple_values=False,
        ),
        MorphologyFeatureSchema(
            name="Multi",
            values=("First", "Second"),
            allows_multiple_values=True,
        ),
    ),
)


def _labels(
    document_id: str,
    sentence_index: int,
    token_count: int,
    *,
    upos_disagreements: tuple[int, ...] = (),
    morphology_disagreements: tuple[int, ...] = (),
) -> SilverSentenceLabels:
    # Labeler prefers UPOS class 1, exclusive label 1, multi value "First",
    # and lemma rule 3 everywhere.
    upos = torch.tensor([[0.1, 0.8, 0.1]]).repeat(token_count, 1).half()
    exclusive = torch.tensor([[0.2, 0.8]]).repeat(token_count, 1).half()
    multi = torch.tensor([[0.9, 0.1]]).repeat(token_count, 1).half()
    agreement_upos = torch.ones(token_count, dtype=torch.int16)
    for index in upos_disagreements:
        agreement_upos[index] = 2
    agreement_exclusive = (
        torch.tensor([[False, True]]).repeat(token_count, 1)
    )
    agreement_multi = (
        torch.tensor([[False, True, False]]).repeat(token_count, 1)
    )
    for index in morphology_disagreements:
        agreement_multi[index] = torch.tensor([True, False, False])
    return SilverSentenceLabels(
        document_id=document_id,
        sentence_index=sentence_index,
        upos_probabilities=upos,
        morphology_probabilities=(exclusive, multi),
        lemma_rule_ids=torch.tensor([[3, 0]]).repeat(token_count, 1).int(),
        lemma_rule_probabilities=(
            torch.tensor([[0.7, 0.2]]).repeat(token_count, 1).half()
        ),
        agreement_upos_ids=agreement_upos,
        agreement_lemma_rule_ids=torch.full((token_count,), 3, dtype=torch.int32),
        agreement_morphology_predictions=(agreement_exclusive, agreement_multi),
    )


def _write_silver_fixture(
    directory: Path,
    records: tuple[SilverSentenceLabels, ...],
    token_counts: tuple[int, ...],
) -> tuple[Path, Path]:
    corpus_path = directory / "pretokenized.jsonl"
    with corpus_path.open("w", encoding="utf-8") as corpus:
        for record, token_count in zip(records, token_counts, strict=True):
            corpus.write(
                json.dumps(
                    {
                        "document_id": record.document_id,
                        "sentence_index": record.sentence_index,
                        "tokens": [f"tok{i}" for i in range(token_count)],
                        "has_space_before": [False]
                        + [True] * (token_count - 1),
                    }
                )
                + "\n"
            )
    labels_directory = directory / "labels"
    reference = write_silver_label_shard(
        records=records,
        path=labels_directory / "labels-00001.pt",
    )
    manifest = SilverLabelManifest(
        format_version=SILVER_LABEL_FORMAT_VERSION,
        corpus_path=str(corpus_path),
        corpus_sha256="0" * 64,
        corpus_manifest={},
        labeler_checkpoint_path="runs/teacher/best.pt",
        labeler_checkpoint_sha256="1" * 64,
        labeler_epoch_index=0,
        calibration={},
        morphology_logit_correction_strength=1.0,
        agreement_checkpoint_path="runs/control/best.pt",
        agreement_checkpoint_sha256="2" * 64,
        agreement_logit_correction_strength=1.0,
        lemma_top_k=2,
        probability_dtype="float16",
        maximum_token_budget=None,
        sentence_count=len(records),
        token_count=sum(token_counts),
        shards=(reference,),
    )
    write_silver_label_manifest(manifest, labels_directory / "labels-manifest.json")
    return corpus_path, labels_directory


def test_silver_loading_applies_agreement_masks_and_discard(
    tmp_path: Path,
) -> None:
    records = (
        _labels("urn:1", 0, 4),
        # One of two tokens disagrees on UPOS: 50% masked > 30% -> discarded.
        _labels("urn:1", 1, 2, upos_disagreements=(0,)),
        # One of four tokens disagrees on morphology: 25% masked -> retained.
        _labels("urn:2", 0, 4, morphology_disagreements=(2,)),
    )
    corpus_path, labels_directory = _write_silver_fixture(
        tmp_path, records, (4, 2, 4)
    )

    sentences, report = load_silver_training_sentences(
        corpus_path=corpus_path,
        labels_directory=labels_directory,
        morphology_schema=MORPHOLOGY_SCHEMA,
        policy=SilverFilterPolicy(),
    )

    assert report.sentence_count == 3
    assert report.retained_sentence_count == 2
    assert report.retained_token_count == 8
    assert sentences[0].morphology_mask.all()
    assert sentences[1].morphology_mask.tolist() == [True, True, False, True]
    assert sentences[1].upos_mask.all()


def test_silver_loading_detects_misaligned_labels(tmp_path: Path) -> None:
    records = (_labels("urn:1", 0, 3),)
    corpus_path, labels_directory = _write_silver_fixture(tmp_path, records, (3,))
    # Corrupt the corpus order by rewriting with a different sentence index.
    lines = corpus_path.read_text().splitlines()
    record = json.loads(lines[0])
    record["sentence_index"] = 7
    corpus_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="not aligned"):
        load_silver_training_sentences(
            corpus_path=corpus_path,
            labels_directory=labels_directory,
            morphology_schema=MORPHOLOGY_SCHEMA,
            policy=SilverFilterPolicy(),
        )


class _StubTokenizer:
    pass


def _tokenized_batch(token_counts: tuple[int, ...]) -> TokenizedBatch:
    batch_size = len(token_counts)
    max_tokens = max(token_counts)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    for row, count in enumerate(token_counts):
        token_mask[row, :count] = True
    subword_count = max_tokens + 2
    return TokenizedBatch(
        input_ids=torch.ones((batch_size, subword_count), dtype=torch.long),
        attention_mask=torch.ones((batch_size, subword_count), dtype=torch.bool),
        first_subword_indices=(
            torch.arange(1, max_tokens + 1).unsqueeze(0).repeat(batch_size, 1)
        ),
        subword_end_indices=(
            torch.arange(2, max_tokens + 2).unsqueeze(0).repeat(batch_size, 1)
        ),
        token_mask=token_mask,
    )


def test_silver_kd_loss_prefers_matching_logits_and_respects_masks() -> None:
    from prism.training import SilverTokenTaskBatch

    token_counts = (3,)
    labels = _labels("urn:1", 0, 3)
    batch = SilverTokenTaskBatch(
        model_inputs=_tokenized_batch(token_counts),
        character_inputs=None,
        upos_probabilities=labels.upos_probabilities.float().unsqueeze(0),
        morphology_probabilities=tuple(
            probabilities.float().unsqueeze(0)
            for probabilities in labels.morphology_probabilities
        ),
        lemma_rule_ids=labels.lemma_rule_ids.long().unsqueeze(0),
        lemma_rule_probabilities=(
            labels.lemma_rule_probabilities.float().unsqueeze(0)
        ),
        upos_mask=torch.ones((1, 3), dtype=torch.bool),
        morphology_mask=torch.ones((1, 3), dtype=torch.bool),
        lemma_mask=torch.ones((1, 3), dtype=torch.bool),
    )

    def logits(bias: float) -> TokenTaskLogits:
        return TokenTaskLogits(
            upos_logits=torch.log(
                torch.tensor([0.1, 0.8, 0.1]) + bias
            ).expand(1, 3, 3),
            morphology_logits=(
                torch.log(torch.tensor([0.2, 0.8]) + bias).expand(1, 3, 2),
                torch.tensor([2.0, -2.0]).expand(1, 3, 2) * (1.0 - bias * 2),
            ),
            lemma_rule_logits=torch.log(
                torch.tensor([0.05, 0.05, 0.1, 0.7, 0.1]) + bias
            ).expand(1, 3, 5),
        )

    matching = compute_silver_kd_loss(
        logits=logits(0.0),
        batch=batch,
        morphology_schema=MORPHOLOGY_SCHEMA,
    )
    mismatching = compute_silver_kd_loss(
        logits=logits(0.9),
        batch=batch,
        morphology_schema=MORPHOLOGY_SCHEMA,
    )
    assert float(matching.total_loss) < float(mismatching.total_loss)

    fully_masked = SilverTokenTaskBatch(
        model_inputs=batch.model_inputs,
        character_inputs=None,
        upos_probabilities=batch.upos_probabilities,
        morphology_probabilities=batch.morphology_probabilities,
        lemma_rule_ids=batch.lemma_rule_ids,
        lemma_rule_probabilities=batch.lemma_rule_probabilities,
        upos_mask=torch.zeros((1, 3), dtype=torch.bool),
        morphology_mask=torch.zeros((1, 3), dtype=torch.bool),
        lemma_mask=torch.zeros((1, 3), dtype=torch.bool),
    )
    masked_losses = compute_silver_kd_loss(
        logits=logits(0.0),
        batch=fully_masked,
        morphology_schema=MORPHOLOGY_SCHEMA,
    )
    assert float(masked_losses.total_loss) == 0.0
    assert masked_losses.upos_token_count == 0


class _TinySilverModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upos = nn.Parameter(torch.zeros(3))
        self.exclusive = nn.Parameter(torch.zeros(2))
        self.multi = nn.Parameter(torch.zeros(2))
        self.lemma_rules = nn.Parameter(torch.zeros(5))

    def forward(self, batch: TokenizedBatch) -> TokenTaskLogits:
        batch_size = batch.token_mask.shape[0]
        token_count = batch.token_mask.shape[1]
        return TokenTaskLogits(
            upos_logits=self.upos.expand(batch_size, token_count, 3),
            morphology_logits=(
                self.exclusive.expand(batch_size, token_count, 2),
                self.multi.expand(batch_size, token_count, 2),
            ),
            lemma_rule_logits=self.lemma_rules.expand(batch_size, token_count, 5),
        )


def test_mixed_epoch_interleaves_gold_and_silver_batches() -> None:
    from prism.data import TokenTaskTargetBatch
    from prism.training import SilverTokenTaskBatch, SupervisedTokenTaskBatch

    model = _TinySilverModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    gold_batch = SupervisedTokenTaskBatch(
        model_inputs=_tokenized_batch((2,)),
        targets=TokenTaskTargetBatch(
            upos_ids=torch.tensor([[1, 2]]),
            morphology_targets=(
                torch.tensor([[[False, True], [True, False]]]),
                torch.tensor(
                    [[[False, True, False], [True, False, False]]]
                ),
            ),
            lemma_rule_ids=torch.tensor([[3, 1]]),
            lemma_rule_mask=torch.tensor([[True, True]]),
            token_mask=torch.tensor([[True, True]]),
        ),
    )
    labels = _labels("urn:1", 0, 2)
    silver_batch = SilverTokenTaskBatch(
        model_inputs=_tokenized_batch((2,)),
        character_inputs=None,
        upos_probabilities=labels.upos_probabilities.float().unsqueeze(0),
        morphology_probabilities=tuple(
            probabilities.float().unsqueeze(0)
            for probabilities in labels.morphology_probabilities
        ),
        lemma_rule_ids=labels.lemma_rule_ids.long().unsqueeze(0),
        lemma_rule_probabilities=(
            labels.lemma_rule_probabilities.float().unsqueeze(0)
        ),
        upos_mask=torch.tensor([[True, False]]),
        morphology_mask=torch.tensor([[True, True]]),
        lemma_mask=torch.tensor([[True, True]]),
    )

    metrics = train_mixed_token_task_epoch(
        student=model,
        teacher=None,
        gold_batches=(gold_batch, gold_batch),
        silver_batches=(silver_batch,),
        gold_batch_count=2,
        silver_batch_count=1,
        order_seed=42,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_gradient_norm=1.0,
        morphology_schema=MORPHOLOGY_SCHEMA,
        silver_loss_weight=0.5,
    )

    assert metrics.silver_batch_count == 1
    assert metrics.silver_token_count == 2
    assert metrics.silver_loss_weight == 0.5
    assert metrics.silver_upos_loss > 0.0
    assert metrics.gold_metrics.batch_count == 2
    assert metrics.silver_total_loss > 0.0


def test_parse_training_arguments_validates_silver_pairs() -> None:
    arguments = parse_training_arguments(
        (
            "--silver-corpus",
            "data/processed/sakspapir-nno/pretokenized.jsonl",
            "--silver-labels",
            "data/processed/sakspapir-nno/labels-pilot-1m",
        )
    )
    assert arguments.silver_loss_weight == 0.5
    assert arguments.silver_require_agreement is True
    assert len(arguments.silver_corpus_paths) == 1

    with pytest.raises(SystemExit):
        parse_training_arguments(
            ("--silver-corpus", "corpus.jsonl")
        )
