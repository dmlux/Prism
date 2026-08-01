import XCTest

@testable import PrismKit

/// End-to-end pipeline validation against decisions recorded from the
/// Python reference tagger on the same frozen artifact.
final class PrismTaggerTests: XCTestCase {
    private func loadTagger() throws -> PrismTagger {
        let artifactURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/prism-no-0.2.2")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: artifactURL.appendingPathComponent("manifest.json").path
            ),
            "Local artifact is not present."
        )
        return try PrismTagger(artifactURL: artifactURL, device: .cpu)
    }

    func testTagsRawTextWithReferenceDecisions() throws {
        let tagger = try loadTagger()

        let sentences = try tagger.tag(text: "Hun kjøpte tre gamle bøker den 17. mai.")

        XCTAssertEqual(sentences.count, 1)
        let tokens = sentences[0].tokens
        XCTAssertEqual(
            tokens.map(\.text),
            ["Hun", "kjøpte", "tre", "gamle", "bøker", "den", "17.", "mai", "."]
        )
        XCTAssertEqual(
            tokens.map(\.upos),
            ["PRON", "VERB", "NUM", "ADJ", "NOUN", "DET", "ADJ", "NOUN", "PUNCT"]
        )
        XCTAssertEqual(
            tokens.map(\.lemma),
            ["hun", "kjøpe", "tre", "gammel", "bok", "den", "17.", "mai", "."]
        )
        XCTAssertEqual(tokens[4].features["Gender"], ["Fem"])
        XCTAssertEqual(tokens[4].features["Number"], ["Plur"])
        XCTAssertTrue(tokens.allSatisfy { $0.uposConfidence > 0.9 })
        XCTAssertTrue(tokens.allSatisfy { $0.lemmaConfidence > 0.9 })
    }

    func testTagsMoreSentencesThanOneBatch() throws {
        let tagger = try loadTagger()
        let sentence = ["Katten", "sov", "."]

        let sentences = try tagger.tag(
            pretokenized: Array(repeating: sentence, count: 11)
        )

        XCTAssertEqual(sentences.count, 11)
        for tagged in sentences {
            XCTAssertEqual(tagged.tokens.map(\.upos), ["NOUN", "VERB", "PUNCT"])
            XCTAssertEqual(tagged.tokens[0].lemma, "katt")
            XCTAssertEqual(tagged.tokens[1].lemma, "sove")
        }
    }
}
