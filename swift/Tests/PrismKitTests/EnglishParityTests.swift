import XCTest

@testable import PrismKit

/// The English artifact uses the ModernBERT/Ettin backbone with a different
/// tokenizer (null unk_token, "[CLS]"/"[SEP]" template, byte-level GPT-2
/// pre-tokenizer, plain NFC) and its own abbreviations, all read from the
/// artifact. PrismKit reproduces the Python reference through the same
/// language-independent code paths it runs for Norwegian. (Engine parity for
/// the same artifact lives in ``EngineParityTests``.)
final class EnglishParityTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// End-to-end oracle: Swift segmentation with the artifact's own
    /// abbreviations plus the ModernBERT byte-level BPE must reproduce the
    /// Python pipeline's subword IDs for every fixture sentence.
    func testEnglishSubwordIdsMatchPythonReference() throws {
        let artifactURL = repositoryRoot.appendingPathComponent("models/prism-en-0.1.0")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: artifactURL.appendingPathComponent("manifest.json").path
            ),
            "Local English artifact is not present."
        )
        let artifact = try PrismArtifact(contentsOf: artifactURL)
        let abbreviations = try XCTUnwrap(artifact.manifest.segmentation?.abbreviations)
        let policy = SegmentationPolicy(
            abbreviationTokens: Set(abbreviations),
            maximumTokenCount: 128
        )
        let tokenizer = try SubwordTokenizer(
            vocabularyURL: artifactURL.appendingPathComponent("vocabulary.json")
        )

        struct Oracle: Decodable {
            let sentenceInputIds: [[Int]]
            enum CodingKeys: String, CodingKey {
                case sentenceInputIds = "sentence_input_ids"
            }
        }
        let text = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "data/examples/harbor-english.txt"
            ),
            encoding: .utf8
        )
        let oracle = try JSONDecoder().decode(
            Oracle.self,
            from: Data(
                contentsOf: repositoryRoot.appendingPathComponent(
                    "data/examples/harbor-english-subword-parity.json"
                )
            )
        )

        let sentences = RuntimeSegmentation.segment(text, policy: policy)
        XCTAssertEqual(sentences.count, oracle.sentenceInputIds.count)
        for (index, sentence) in sentences.enumerated() {
            XCTAssertEqual(
                tokenizer.encode(sentence).inputIds,
                oracle.sentenceInputIds[index],
                "harbor-english sentence \(index): \(sentence.tokens)"
            )
        }
    }
}
