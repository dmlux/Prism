import pytest
import torch

from prism.exporting import (
    ARTIFACT_MANIFEST_FORMAT_VERSION,
    BackboneProvenance,
    CheckpointProvenance,
    FixedExportShapes,
    ModelArtifactManifest,
    ModelDataFileEntry,
    ModelProgramEntry,
    ParityFixtureBatch,
    TensorSpec,
    TokenizerContract,
    TreebankProvenance,
    build_fixtures_payload,
    build_labels_payload,
    decoded_sentence_predictions,
    deserialize_tensor,
    maximum_task_probability_difference,
    serialize_tensor,
    top_k_output_tensors,
)
from prism.modeling.outputs import TokenTaskLogits
from prism.schema import (
    LemmaEditRule,
    LemmaRuleSchema,
    MorphologyFeatureSchema,
    MorphologySchema,
    TokenTaskSchema,
    UposSchema,
)


def _schema() -> TokenTaskSchema:
    return TokenTaskSchema(
        upos=UposSchema(version=1, labels=("NOUN", "VERB")),
        morphology=MorphologySchema(
            version=1,
            features=(
                MorphologyFeatureSchema(
                    name="Gender",
                    values=("Fem", "Masc"),
                    allows_multiple_values=True,
                ),
                MorphologyFeatureSchema(
                    name="Number",
                    values=("Plur", "Sing"),
                    allows_multiple_values=False,
                ),
            ),
        ),
        lemma_rules=LemmaRuleSchema(
            version=1,
            rules=(
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=0,
                    prefix_addition="",
                    suffix_addition="",
                ),
                LemmaEditRule(
                    prefix_removal=0,
                    suffix_removal=2,
                    prefix_addition="",
                    suffix_addition="e",
                ),
            ),
        ),
    )


def _manifest() -> ModelArtifactManifest:
    return ModelArtifactManifest(
        artifact_name="prism-no",
        artifact_version="0.1.0",
        language_tags=("nb", "nn"),
        tasks=("upos", "morphology", "lemma_rules"),
        labels_file="labels.json",
        vocabulary_file="vocabulary.json",
        fixtures_file="fixtures.json",
        character_unicode_normalization="NFC",
        tokenizer=TokenizerContract(
            file_name="vocabulary.json",
            class_name="PreTrainedTokenizerFast",
            padding_token_id=3,
        ),
        programs=(
            ModelProgramEntry(
                file_name="model-xnnpack.pte",
                format="executorch-pte",
                backend="xnnpack",
                precision="fp32",
                sha256="ab" * 32,
                size_bytes=1024,
                shapes=FixedExportShapes(
                    batch_size=8,
                    subword_count=160,
                    token_count=96,
                    character_count=32,
                ),
                inputs=(
                    TensorSpec(name="input_ids", dtype="int64", shape=(8, 160)),
                    TensorSpec(name="token_mask", dtype="bool", shape=(8, 96)),
                ),
                output_names=("upos_logits", "lemma_rule_logits"),
                parity_maximum_probability_difference=1.5e-5,
            ),
        ),
        checkpoint=CheckpointProvenance(
            run_name="example-run",
            file_name="best.pt",
            epoch_index=9,
            sha256="cd" * 32,
            morphology_logit_correction_strength=0.25,
        ),
        backbone=BackboneProvenance(
            model_id="ltg/norbert4-xsmall",
            revision="7483327d36a2daa5dbe936c68aa277149c6f9632",
        ),
        treebanks=(
            TreebankProvenance(
                repository_id="UniversalDependencies/UD_Norwegian-Bokmaal",
                revision="396d11f0c2bd290a2a2711015c04ac25bc3dcc06",
                license_id="CC-BY-SA-4.0",
            ),
        ),
        quantization="none",
        calibration_file=None,
    )


def test_manifest_serializes_the_complete_contract() -> None:
    payload = _manifest().to_json_dict()

    assert payload["manifest_format_version"] == ARTIFACT_MANIFEST_FORMAT_VERSION
    assert payload["artifact_name"] == "prism-no"
    assert payload["language_tags"] == ["nb", "nn"]
    assert payload["tokenizer"] == {
        "file_name": "vocabulary.json",
        "class_name": "PreTrainedTokenizerFast",
        "padding_token_id": 3,
    }

    program = payload["programs"][0]
    assert program["shapes"] == {
        "batch_size": 8,
        "subword_count": 160,
        "token_count": 96,
        "character_count": 32,
    }
    assert program["inputs"][0] == {
        "name": "input_ids",
        "dtype": "int64",
        "shape": [8, 160],
    }
    assert payload["checkpoint"]["morphology_logit_correction_strength"] == 0.25
    assert payload["calibration_file"] is None


def test_manifest_rejects_empty_sections() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError):
        ModelArtifactManifest(
            artifact_name="prism-no",
            artifact_version="0.1.0",
            language_tags=(),
            tasks=manifest.tasks,
            labels_file=manifest.labels_file,
            vocabulary_file=manifest.vocabulary_file,
            fixtures_file=manifest.fixtures_file,
            character_unicode_normalization="NFC",
            tokenizer=manifest.tokenizer,
            programs=manifest.programs,
            checkpoint=manifest.checkpoint,
            backbone=manifest.backbone,
            treebanks=manifest.treebanks,
            quantization="none",
            calibration_file=None,
        )


def test_manifest_requires_entries_for_program_data_files() -> None:
    manifest = _manifest()
    program = manifest.programs[0]
    separated = ModelProgramEntry(
        file_name=program.file_name,
        format=program.format,
        backend=program.backend,
        precision=program.precision,
        sha256=program.sha256,
        size_bytes=program.size_bytes,
        shapes=program.shapes,
        inputs=program.inputs,
        output_names=program.output_names,
        parity_maximum_probability_difference=(
            program.parity_maximum_probability_difference
        ),
        data_files=("model.ptd",),
    )

    with pytest.raises(ValueError, match="model.ptd"):
        ModelArtifactManifest(
            artifact_name=manifest.artifact_name,
            artifact_version=manifest.artifact_version,
            language_tags=manifest.language_tags,
            tasks=manifest.tasks,
            labels_file=manifest.labels_file,
            vocabulary_file=manifest.vocabulary_file,
            fixtures_file=manifest.fixtures_file,
            character_unicode_normalization="NFC",
            tokenizer=manifest.tokenizer,
            programs=(separated,),
            checkpoint=manifest.checkpoint,
            backbone=manifest.backbone,
            treebanks=manifest.treebanks,
            quantization="none",
            calibration_file=None,
        )

    complete = ModelArtifactManifest(
        artifact_name=manifest.artifact_name,
        artifact_version=manifest.artifact_version,
        language_tags=manifest.language_tags,
        tasks=manifest.tasks,
        labels_file=manifest.labels_file,
        vocabulary_file=manifest.vocabulary_file,
        fixtures_file=manifest.fixtures_file,
        character_unicode_normalization="NFC",
        tokenizer=manifest.tokenizer,
        programs=(separated,),
        data_files=(
            ModelDataFileEntry(
                file_name="model.ptd",
                sha256="ef" * 32,
                size_bytes=2048,
            ),
        ),
        checkpoint=manifest.checkpoint,
        backbone=manifest.backbone,
        treebanks=manifest.treebanks,
        quantization="none",
        calibration_file=None,
    )
    payload = complete.to_json_dict()
    assert payload["programs"][0]["data_files"] == ["model.ptd"]
    assert payload["data_files"][0]["file_name"] == "model.ptd"


def test_labels_payload_serializes_schema_and_characters() -> None:
    payload = build_labels_payload(
        schema=_schema(),
        character_vocabulary=None,
        maximum_character_count=None,
    )

    assert payload["schema"]["upos"]["labels"] == ["NOUN", "VERB"]
    assert payload["character_vocabulary"] is None

    with pytest.raises(ValueError):
        build_labels_payload(
            schema=_schema(),
            character_vocabulary=None,
            maximum_character_count=32,
        )


def test_tensor_serialization_round_trip() -> None:
    tensor = torch.tensor([[1.5, -2.0], [0.0, 3.25]], dtype=torch.float32)

    name, restored = deserialize_tensor(serialize_tensor("upos_logits", tensor))

    assert name == "upos_logits"
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, tensor)


def test_fixtures_payload_contains_batches() -> None:
    fixture = ParityFixtureBatch(
        name="nb-development",
        input_tensors=(("input_ids", torch.tensor([[1, 2]], dtype=torch.long)),),
        expected_output_tensors=(("upos_logits", torch.tensor([[[0.5, -0.5]]])),),
        decoded_sentences=({"tokens": []},),
    )

    payload = build_fixtures_payload((fixture,), probability_tolerance=5e-3)

    assert payload["fixtures"][0]["name"] == "nb-development"
    assert payload["fixtures"][0]["inputs"][0]["dtype"] == "int64"
    assert payload["comparison"]["space"] == "task-probabilities"
    assert payload["comparison"]["probability_tolerance"] == 5e-3

    with pytest.raises(ValueError):
        build_fixtures_payload((), probability_tolerance=5e-3)
    with pytest.raises(ValueError):
        build_fixtures_payload((fixture,), probability_tolerance=0.0)


def test_top_k_output_tensors_records_ids_and_values() -> None:
    logits = torch.tensor([[[0.5, 3.0, -1.0, 2.0]]])

    (ids_name, ids), (values_name, values) = top_k_output_tensors(
        name="lemma_rule_logits",
        logits=logits,
        top_k=2,
    )

    assert ids_name == "lemma_rule_logits:top_ids"
    assert values_name == "lemma_rule_logits:top_values"
    assert ids[0, 0].tolist() == [1, 3]
    assert values[0, 0].tolist() == [3.0, 2.0]

    with pytest.raises(ValueError):
        top_k_output_tensors(name="lemma_rule_logits", logits=logits, top_k=0)


def test_maximum_task_probability_difference_ignores_saturated_logit_gaps() -> None:
    schema = _schema()
    token_mask = torch.tensor([[True, False]])
    reference = (
        torch.tensor([[[4.0, -4.0], [0.0, 0.0]]]),
        torch.tensor([[[14.0, -14.0], [0.0, 0.0]]]),
        torch.tensor([[[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]]),
        torch.tensor([[[0.0, 2.0], [0.0, 0.0]]]),
    )
    # A large logit gap on a saturated sigmoid and garbage at the padded
    # position must both stay invisible in probability space.
    candidate = (
        reference[0],
        torch.tensor([[[14.25, -14.25], [9.0, -9.0]]]),
        reference[2],
        reference[3],
    )

    difference = maximum_task_probability_difference(
        schema=schema,
        reference_outputs=reference,
        candidate_outputs=candidate,
        token_mask=token_mask,
    )

    assert difference < 1e-6


def test_maximum_task_probability_difference_reports_real_shifts() -> None:
    schema = _schema()
    token_mask = torch.tensor([[True]])
    reference = (
        torch.tensor([[[1.0, -1.0]]]),
        torch.tensor([[[0.0, 0.0]]]),
        torch.tensor([[[0.0, 0.0, 3.0]]]),
        torch.tensor([[[0.0, 2.0]]]),
    )
    candidate = (
        torch.tensor([[[-1.0, 1.0]]]),
        reference[1],
        reference[2],
        reference[3],
    )

    difference = maximum_task_probability_difference(
        schema=schema,
        reference_outputs=reference,
        candidate_outputs=candidate,
        token_mask=token_mask,
    )

    assert difference > 0.5


def test_decoded_sentence_predictions_reports_labels_and_lemmas() -> None:
    schema = _schema()
    token_mask = torch.tensor([[True, True]])
    logits = TokenTaskLogits(
        upos_logits=torch.tensor([[[2.0, -1.0], [-1.0, 2.0]]]),
        morphology_logits=(
            # Multi-value Gender: both active on token 0, none on token 1.
            torch.tensor([[[1.0, 2.0], [-1.0, -2.0]]]),
            # Categorical Number over (<NONE>, Plur, Sing).
            torch.tensor([[[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]]),
        ),
        lemma_rule_logits=torch.tensor([[[0.0, 2.0], [2.0, 0.0]]]),
    )

    decoded = decoded_sentence_predictions(
        schema=schema,
        logits=logits,
        token_mask=token_mask,
        sentence_tokens=(("husene", "gikk"),),
    )

    first, second = decoded[0]["tokens"]
    assert first["upos"] == "NOUN"
    assert first["morphology"] == {"Gender": ["Fem", "Masc"], "Number": ["Sing"]}
    assert first["lemma"] == "husee"
    assert second["upos"] == "VERB"
    assert second["morphology"] == {"Gender": [], "Number": []}
    assert second["lemma"] == "gikk"


def test_decoded_sentence_predictions_flags_invalid_lemma_rules() -> None:
    schema = _schema()
    logits = TokenTaskLogits(
        upos_logits=torch.tensor([[[2.0, -1.0]]]),
        morphology_logits=(
            torch.tensor([[[-1.0, -1.0]]]),
            torch.tensor([[[3.0, 0.0, 0.0]]]),
        ),
        lemma_rule_logits=torch.tensor([[[0.0, 2.0]]]),
    )

    decoded = decoded_sentence_predictions(
        schema=schema,
        logits=logits,
        token_mask=torch.tensor([[True]]),
        sentence_tokens=(("å",),),
    )

    assert decoded[0]["tokens"][0]["lemma"] == "<INVALID_LEMMA_RULE>"
