import XCTest

@testable import PrismKit

/// Token-by-token parity against the reference Hugging Face tokenizer,
/// recorded in Resources/subword-parity.json by the Python exporter.
final class SubwordTokenizerTests: XCTestCase {
    private struct ParityCase: Decodable {
        let tokens: [String]
        let hasSpaceBefore: [Bool]
        let inputIds: [Int]
        let firstSubwordIndices: [Int]
        let subwordEndIndices: [Int]
        enum CodingKeys: String, CodingKey {
            case tokens
            case hasSpaceBefore = "has_space_before"
            case inputIds = "input_ids"
            case firstSubwordIndices = "first_subword_indices"
            case subwordEndIndices = "subword_end_indices"
        }
    }
    private struct ParityFile: Decodable { let cases: [ParityCase] }

    func testMatchesReferenceTokenizer() throws {
        let vocabularyURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/prism-no-0.2.1/vocabulary.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: vocabularyURL.path),
            "Local artifact is not present."
        )
        let tokenizer = try SubwordTokenizer(vocabularyURL: vocabularyURL)

        let parityURL = try XCTUnwrap(
            Bundle.module.url(forResource: "subword-parity", withExtension: "json")
        )
        let parity = try JSONDecoder().decode(
            ParityFile.self,
            from: Data(contentsOf: parityURL)
        )
        XCTAssertFalse(parity.cases.isEmpty)

        for (index, expected) in parity.cases.enumerated() {
            let encoded = tokenizer.encode(
                PretokenizedSentence(
                    tokens: expected.tokens,
                    hasSpaceBefore: expected.hasSpaceBefore
                )
            )
            XCTAssertEqual(
                encoded.inputIds, expected.inputIds,
                "input_ids mismatch in case \(index): \(expected.tokens)"
            )
            XCTAssertEqual(
                encoded.firstSubwordIndices, expected.firstSubwordIndices,
                "first_subword_indices mismatch in case \(index)"
            )
            XCTAssertEqual(
                encoded.subwordEndIndices, expected.subwordEndIndices,
                "subword_end_indices mismatch in case \(index)"
            )
        }
    }
}
