import Foundation
import Testing

@testable import PrismKit

/// Token-by-token parity against the reference Hugging Face tokenizer,
/// recorded in Resources/subword-parity.json by the Python exporter.
struct SubwordTokenizerTests {
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

    static var vocabularyURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/prism-no-0.2.5/vocabulary.json")
    }

    static var vocabularyPresent: Bool {
        FileManager.default.fileExists(atPath: vocabularyURL.path)
    }

    @Test(.enabled(if: SubwordTokenizerTests.vocabularyPresent, "Local artifact is not present."))
    func matchesReferenceTokenizer() throws {
        let tokenizer = try SubwordTokenizer(vocabularyURL: Self.vocabularyURL)

        let parityURL = try #require(
            Bundle.module.url(forResource: "subword-parity", withExtension: "json")
        )
        let parity = try JSONDecoder().decode(
            ParityFile.self,
            from: Data(contentsOf: parityURL)
        )
        #expect(!parity.cases.isEmpty)

        for (index, expected) in parity.cases.enumerated() {
            let encoded = tokenizer.encode(
                PretokenizedSentence(
                    tokens: expected.tokens,
                    hasSpaceBefore: expected.hasSpaceBefore
                )
            )
            #expect(
                encoded.inputIds == expected.inputIds,
                "input_ids mismatch in case \(index): \(expected.tokens)"
            )
            #expect(
                encoded.firstSubwordIndices == expected.firstSubwordIndices,
                "first_subword_indices mismatch in case \(index)"
            )
            #expect(
                encoded.subwordEndIndices == expected.subwordEndIndices,
                "subword_end_indices mismatch in case \(index)"
            )
        }
    }
}
