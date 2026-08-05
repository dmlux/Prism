import XCTest

@testable import PrismKit

/// Cross-language parity against the Python reference implementation,
/// measured on the checked-in CC0 example texts (data/examples/README.md).
final class ChapterParityTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testFixturesMatchPythonReferenceCounts() throws {
        let expectations: [(fixture: String, sentences: Int, tokens: Int)] = [
            ("skarvholmen-bokmaal", 55, 905),
            ("fjellvatnet-nynorsk", 41, 803),
        ]
        for expected in expectations {
            let textURL = repositoryRoot.appendingPathComponent(
                "data/examples/\(expected.fixture).txt"
            )
            let text = try String(contentsOf: textURL, encoding: .utf8)

            let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

            XCTAssertEqual(sentences.count, expected.sentences, expected.fixture)
            XCTAssertEqual(
                sentences.reduce(0) { $0 + $1.tokens.count },
                expected.tokens,
                expected.fixture
            )
        }
    }

    /// End-to-end oracle: Swift segmentation plus Swift BPE must reproduce
    /// the Python pipeline's subword IDs for every fixture sentence. A
    /// mismatch in either layer surfaces as an ID difference.
    func testFixtureSubwordIdsMatchPythonReference() throws {
        let vocabularyURL = repositoryRoot.appendingPathComponent(
            "models/prism-no-0.2.3/vocabulary.json"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: vocabularyURL.path),
            "Local artifact is not present."
        )
        let tokenizer = try SubwordTokenizer(vocabularyURL: vocabularyURL)

        struct Oracle: Decodable {
            let sentenceInputIds: [[Int]]
            enum CodingKeys: String, CodingKey {
                case sentenceInputIds = "sentence_input_ids"
            }
        }

        for fixture in ["skarvholmen-bokmaal", "fjellvatnet-nynorsk"] {
            let textURL = repositoryRoot.appendingPathComponent(
                "data/examples/\(fixture).txt"
            )
            let oracleURL = repositoryRoot.appendingPathComponent(
                "data/examples/\(fixture)-subword-parity.json"
            )
            let text = try String(contentsOf: textURL, encoding: .utf8)
            let oracle = try JSONDecoder().decode(
                Oracle.self,
                from: Data(contentsOf: oracleURL)
            )

            let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

            XCTAssertEqual(sentences.count, oracle.sentenceInputIds.count, fixture)
            for (index, sentence) in sentences.enumerated() {
                XCTAssertEqual(
                    tokenizer.encode(sentence).inputIds,
                    oracle.sentenceInputIds[index],
                    "\(fixture) sentence \(index): \(sentence.tokens)"
                )
            }
        }
    }
}
