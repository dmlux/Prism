import XCTest

@testable import PrismKit

/// Cross-language parity against the Python reference implementation,
/// measured on the untracked local book-chapter fixture when present.
final class ChapterParityTests: XCTestCase {
    func testChapterMatchesPythonReferenceCounts() throws {
        let chapterURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("data/examples/hp7kap1.txt")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: chapterURL.path),
            "Local chapter fixture is not present."
        )

        let text = try String(contentsOf: chapterURL, encoding: .utf8)
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 247)
        XCTAssertEqual(sentences.reduce(0) { $0 + $1.tokens.count }, 3783)
    }

    /// End-to-end oracle: Swift segmentation plus Swift BPE must reproduce
    /// the Python pipeline's subword IDs for every chapter sentence. A
    /// mismatch in either layer surfaces as an ID difference.
    func testChapterSubwordIdsMatchPythonReference() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let chapterURL = root.appendingPathComponent("data/examples/hp7kap1.txt")
        let oracleURL = root.appendingPathComponent(
            "data/examples/hp7kap1-subword-parity.json"
        )
        let vocabularyURL = root.appendingPathComponent(
            "models/prism-no-0.2.2/vocabulary.json"
        )
        for url in [chapterURL, oracleURL, vocabularyURL] {
            try XCTSkipUnless(
                FileManager.default.fileExists(atPath: url.path),
                "Local fixture is not present."
            )
        }

        struct Oracle: Decodable {
            let sentenceInputIds: [[Int]]
            enum CodingKeys: String, CodingKey {
                case sentenceInputIds = "sentence_input_ids"
            }
        }
        let oracle = try JSONDecoder().decode(
            Oracle.self,
            from: Data(contentsOf: oracleURL)
        )

        let text = try String(contentsOf: chapterURL, encoding: .utf8)
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())
        let tokenizer = try SubwordTokenizer(vocabularyURL: vocabularyURL)

        XCTAssertEqual(sentences.count, oracle.sentenceInputIds.count)
        let started = Date()
        for (index, sentence) in sentences.enumerated() {
            let encoded = tokenizer.encode(sentence)
            XCTAssertEqual(
                encoded.inputIds,
                oracle.sentenceInputIds[index],
                "Subword IDs diverge in sentence \(index): \(sentence.tokens)"
            )
        }
        let elapsed = Date().timeIntervalSince(started)
        print("PrismKit BPE: \(sentences.count) sentences in \(Int(elapsed * 1000)) ms")
    }
}
