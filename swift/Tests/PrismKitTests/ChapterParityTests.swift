import Foundation
import Testing

@testable import PrismKit

/// Cross-language parity against the Python reference implementation,
/// measured on the checked-in CC0 example texts (data/examples/README.md).
struct ChapterParityTests {
    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static var vocabularyURL: URL {
        repositoryRoot.appendingPathComponent("models/prism-no-0.2.4/vocabulary.json")
    }

    static var vocabularyPresent: Bool {
        FileManager.default.fileExists(atPath: vocabularyURL.path)
    }

    @Test func fixturesMatchPythonReferenceCounts() throws {
        let expectations: [(fixture: String, sentences: Int, tokens: Int)] = [
            ("skarvholmen-bokmaal", 55, 905),
            ("fjellvatnet-nynorsk", 41, 803),
        ]
        for expected in expectations {
            let textURL = Self.repositoryRoot.appendingPathComponent(
                "data/examples/\(expected.fixture).txt"
            )
            let text = try String(contentsOf: textURL, encoding: .utf8)

            let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

            #expect(sentences.count == expected.sentences, "\(expected.fixture)")
            #expect(
                sentences.reduce(0) { $0 + $1.tokens.count } == expected.tokens,
                "\(expected.fixture)"
            )
        }
    }

    /// End-to-end oracle: Swift segmentation plus Swift BPE must reproduce
    /// the Python pipeline's subword IDs for every fixture sentence. A
    /// mismatch in either layer surfaces as an ID difference.
    @Test(.enabled(if: ChapterParityTests.vocabularyPresent, "Local artifact is not present."))
    func fixtureSubwordIdsMatchPythonReference() throws {
        let tokenizer = try SubwordTokenizer(vocabularyURL: Self.vocabularyURL)

        struct Oracle: Decodable {
            let sentenceInputIds: [[Int]]
            enum CodingKeys: String, CodingKey {
                case sentenceInputIds = "sentence_input_ids"
            }
        }

        for fixture in ["skarvholmen-bokmaal", "fjellvatnet-nynorsk"] {
            let textURL = Self.repositoryRoot.appendingPathComponent(
                "data/examples/\(fixture).txt"
            )
            let oracleURL = Self.repositoryRoot.appendingPathComponent(
                "data/examples/\(fixture)-subword-parity.json"
            )
            let text = try String(contentsOf: textURL, encoding: .utf8)
            let oracle = try JSONDecoder().decode(
                Oracle.self,
                from: Data(contentsOf: oracleURL)
            )

            let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

            #expect(sentences.count == oracle.sentenceInputIds.count, "\(fixture)")
            for (index, sentence) in sentences.enumerated() {
                #expect(
                    tokenizer.encode(sentence).inputIds == oracle.sentenceInputIds[index],
                    "\(fixture) sentence \(index): \(sentence.tokens)"
                )
            }
        }
    }
}
