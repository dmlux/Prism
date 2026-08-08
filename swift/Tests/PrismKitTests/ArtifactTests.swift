import Foundation
import Testing

@testable import PrismKit

struct ArtifactTests {
    private func writeArtifact(manifest: String, labels: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try manifest.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("manifest.json")
        )
        try labels.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("labels.json")
        )
        return directory
    }

    private let manifestJSON = """
        {
          "manifest_format_version": 1,
          "artifact_name": "prism-no",
          "artifact_version": "0.2.0",
          "language_tags": ["nb", "nn"],
          "labels_file": "labels.json",
          "vocabulary_file": "vocabulary.json",
          "character_unicode_normalization": "NFC",
          "tokenizer": {
            "file_name": "vocabulary.json",
            "class_name": "TokenizersBackend",
            "padding_token_id": 3
          },
          "programs": [
            {
              "file_name": "model-xnnpack.pte",
              "format": "executorch-pte",
              "backend": "xnnpack",
              "precision": "fp32",
              "sha256": "00",
              "size_bytes": 42,
              "shapes": {
                "batch_size": 8,
                "subword_count": 160,
                "token_count": 96,
                "character_count": 32
              },
              "output_names": ["upos_probabilities"],
              "data_files": ["model.ptd"]
            }
          ],
          "data_files": [
            {"file_name": "model.ptd", "sha256": "00", "size_bytes": 7}
          ],
          "calibration_file": "calibration.json"
        }
        """

    private let labelsJSON = """
        {
          "labels_format_version": 1,
          "schema": {
            "format_version": 1,
            "upos": {"version": 1, "labels": ["ADJ", "NOUN"]},
            "morphology": {
              "version": 1,
              "features": [
                {"name": "Gender", "values": ["Fem", "Masc", "Neut"],
                 "allows_multiple_values": false}
              ]
            },
            "lemma_rules": {
              "version": 1,
              "rules": [
                {"prefix_removal": 0, "suffix_removal": 0,
                 "prefix_addition": "", "suffix_addition": ""},
                {"prefix_removal": 0, "suffix_removal": 2,
                 "prefix_addition": "", "suffix_addition": ""}
              ]
            }
          },
          "character_vocabulary": {"version": 1, "characters": ["<PAD>", "a", "b"]},
          "maximum_character_count": 32
        }
        """

    @Test func decodesManifestAndLabels() throws {
        let directory = try writeArtifact(manifest: manifestJSON, labels: labelsJSON)
        let artifact = try PrismArtifact(contentsOf: directory)

        #expect(artifact.manifest.artifactVersion == "0.2.0")
        #expect(artifact.manifest.tokenizer.paddingTokenId == 3)
        #expect(artifact.manifest.calibrationFile == "calibration.json")
        #expect(artifact.manifest.programs[0].dataFiles == ["model.ptd"])
        #expect(artifact.manifest.dataFiles?.first?.fileName == "model.ptd")
        #expect(artifact.labels.schema.upos.labels == ["ADJ", "NOUN"])
        #expect(artifact.labels.schema.morphology.features[0].name == "Gender")
        #expect(!artifact.labels.schema.morphology.features[0].allowsMultipleValues)
        #expect(artifact.labels.characterVocabulary?.identifiers()["b"] == 2)
    }

    @Test func selectsProgramByDevice() throws {
        let directory = try writeArtifact(manifest: manifestJSON, labels: labelsJSON)
        let artifact = try PrismArtifact(contentsOf: directory)

        #expect(try artifact.program(for: .cpu).backend == "xnnpack")
        #expect(try artifact.program(for: .automatic).backend == "xnnpack")
        #expect(throws: PrismError.deviceUnavailable(.gpu)) {
            try artifact.program(for: .gpu)
        }
    }

    @Test func lemmaEditRuleMirrorsReferenceSemantics() throws {
        let identity = LemmaEditRule(
            prefixRemoval: 0, suffixRemoval: 0, prefixAddition: "", suffixAddition: ""
        )
        #expect(try identity.apply(to: "bøker") == "bøker")

        let pluralToStem = LemmaEditRule(
            prefixRemoval: 0, suffixRemoval: 2, prefixAddition: "", suffixAddition: ""
        )
        #expect(try pluralToStem.apply(to: "bøker") == "bøk")

        let overRemoval = LemmaEditRule(
            prefixRemoval: 3, suffixRemoval: 3, prefixAddition: "", suffixAddition: ""
        )
        #expect(throws: (any Error).self) {
            try overRemoval.apply(to: "abc")
        }
    }

    @Test func missingManifestSurfacesTypedError() {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        #expect(throws: PrismError.missingArtifactFile("manifest.json")) {
            try PrismArtifact(contentsOf: directory)
        }
    }

    @Test func missingMetadataFailsLoudly() throws {
        // A manifest without the required language tags must fail to decode,
        // not be guessed from the directory name.
        let manifest = manifestJSON.replacingOccurrences(
            of: "\"language_tags\": [\"nb\", \"nn\"],", with: ""
        )
        let directory = try writeArtifact(manifest: manifest, labels: labelsJSON)
        #expect(throws: (any Error).self) {
            try PrismArtifact(contentsOf: directory)
        }
    }

    @Test func invalidLanguageTagsFailLoudly() throws {
        // Language tags must be a list; a bare string is a hard decode error.
        let manifest = manifestJSON.replacingOccurrences(
            of: "[\"nb\", \"nn\"]", with: "\"nb\""
        )
        let directory = try writeArtifact(manifest: manifest, labels: labelsJSON)
        #expect(throws: (any Error).self) {
            try PrismArtifact(contentsOf: directory)
        }
    }
}
