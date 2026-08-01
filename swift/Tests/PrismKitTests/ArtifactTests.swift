import XCTest

@testable import PrismKit

final class ArtifactTests: XCTestCase {
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
              "output_names": ["upos_probabilities"]
            }
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

    func testDecodesManifestAndLabels() throws {
        let directory = try writeArtifact(manifest: manifestJSON, labels: labelsJSON)
        let artifact = try PrismArtifact(contentsOf: directory)

        XCTAssertEqual(artifact.manifest.artifactVersion, "0.2.0")
        XCTAssertEqual(artifact.manifest.tokenizer.paddingTokenId, 3)
        XCTAssertEqual(artifact.manifest.calibrationFile, "calibration.json")
        XCTAssertEqual(artifact.labels.schema.upos.labels, ["ADJ", "NOUN"])
        XCTAssertEqual(artifact.labels.schema.morphology.features[0].name, "Gender")
        XCTAssertFalse(artifact.labels.schema.morphology.features[0].allowsMultipleValues)
        XCTAssertEqual(artifact.labels.characterVocabulary?.identifiers()["b"], 2)
    }

    func testSelectsProgramByDevice() throws {
        let directory = try writeArtifact(manifest: manifestJSON, labels: labelsJSON)
        let artifact = try PrismArtifact(contentsOf: directory)

        XCTAssertEqual(try artifact.program(for: .cpu).backend, "xnnpack")
        XCTAssertEqual(try artifact.program(for: .automatic).backend, "xnnpack")
        XCTAssertThrowsError(try artifact.program(for: .gpu)) { error in
            XCTAssertEqual(error as? PrismError, .deviceUnavailable(.gpu))
        }
    }

    func testLemmaEditRuleMirrorsReferenceSemantics() throws {
        let identity = LemmaEditRule(
            prefixRemoval: 0, suffixRemoval: 0, prefixAddition: "", suffixAddition: ""
        )
        XCTAssertEqual(try identity.apply(to: "bøker"), "bøker")

        let pluralToStem = LemmaEditRule(
            prefixRemoval: 0, suffixRemoval: 2, prefixAddition: "", suffixAddition: ""
        )
        XCTAssertEqual(try pluralToStem.apply(to: "bøker"), "bøk")

        let overRemoval = LemmaEditRule(
            prefixRemoval: 3, suffixRemoval: 3, prefixAddition: "", suffixAddition: ""
        )
        XCTAssertThrowsError(try overRemoval.apply(to: "abc"))
    }

    func testMissingManifestSurfacesTypedError() {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try PrismArtifact(contentsOf: directory)) { error in
            XCTAssertEqual(error as? PrismError, .missingArtifactFile("manifest.json"))
        }
    }
}
