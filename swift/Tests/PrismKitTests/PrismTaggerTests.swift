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
            .appendingPathComponent("models/prism-no-0.2.3")
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

    func testRawTextResultsCarrySourceRanges() throws {
        let tagger = try loadTagger()
        let text = "Hun kjøpte tre gamle bøker den 17. mai."

        let sentences = try tagger.tag(text: text)

        XCTAssertEqual(sentences.count, 1)
        // Byte offsets shared with the C++ and Java suites (parity).
        XCTAssertEqual(
            sentences[0].tokens.map(\.sourceRanges),
            [
                [Utf8ByteRange(start: 0, end: 3)],
                [Utf8ByteRange(start: 4, end: 11)],
                [Utf8ByteRange(start: 12, end: 15)],
                [Utf8ByteRange(start: 16, end: 21)],
                [Utf8ByteRange(start: 22, end: 28)],
                [Utf8ByteRange(start: 29, end: 32)],
                [Utf8ByteRange(start: 33, end: 36)],
                [Utf8ByteRange(start: 37, end: 40)],
                [Utf8ByteRange(start: 40, end: 41)],
            ]
        )
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 41)])
        let boker = sentences[0].tokens[4].sourceRanges[0].range(in: text)!
        XCTAssertEqual(String(text[boker]), "bøker")
    }

    func testPretokenizedInputCarriesNoSourceRanges() throws {
        let tagger = try loadTagger()

        let sentences = try tagger.tag(pretokenized: [["Katten", "sov", "."]])

        XCTAssertEqual(sentences.count, 1)
        XCTAssertTrue(sentences[0].sourceRanges.isEmpty)
        XCTAssertTrue(sentences[0].tokens.allSatisfy(\.sourceRanges.isEmpty))
    }

    func testCallerProvidedSourceRangesPassThrough() throws {
        let tagger = try loadTagger()
        let sentence = PretokenizedSentence(
            tokens: ["Katten", "sov", "."],
            hasSpaceBefore: [false, true, false],
            tokenSourceRanges: [
                [Utf8ByteRange(start: 0, end: 6)],
                [Utf8ByteRange(start: 7, end: 10)],
                [Utf8ByteRange(start: 10, end: 11)],
            ],
            sourceRanges: [Utf8ByteRange(start: 0, end: 11)]
        )

        let sentences = try tagger.tag(sentences: [sentence])

        XCTAssertEqual(sentences.count, 1)
        XCTAssertEqual(sentences[0].sourceRanges, sentence.sourceRanges)
        XCTAssertEqual(
            sentences[0].tokens.map(\.sourceRanges), sentence.tokenSourceRanges
        )
    }

    func testExposesArtifactMetadata() throws {
        let tagger = try loadTagger()

        XCTAssertEqual(tagger.artifactName, "prism-no")
        XCTAssertEqual(tagger.artifactVersion, "0.2.3")
        // Since 0.2.3 the manifest also declares the BCP 47 macrolanguage,
        // so plain-"no" documents match without host-side aliases.
        XCTAssertEqual(tagger.languageTags, ["nb", "nn", "no"])

        // Label inventories mirrored from labels.json.
        XCTAssertEqual(tagger.uposLabels.count, 17)
        XCTAssertTrue(tagger.uposLabels.contains("NOUN"))
        XCTAssertEqual(tagger.morphologyFeatures.count, 18)
        XCTAssertTrue(
            tagger.morphologyFeatures.first { $0.name == "Number" }?
                .values.contains("Plur") ?? false
        )
    }

    func testReportsTheUposDistributionPerToken() throws {
        let tagger = try loadTagger()

        let sentences = try tagger.tag(pretokenized: [["Katten", "sov", "."]])
        for token in sentences[0].tokens {
            let distribution = token.uposDistribution
            // One entry per artifact UPOS label, sorted by descending
            // probability; the first entry is the reported decision.
            XCTAssertEqual(distribution.count, tagger.uposLabels.count)
            XCTAssertEqual(distribution[0].upos, token.upos)
            XCTAssertEqual(distribution[0].probability, token.uposConfidence)
            for entry in 1..<distribution.count {
                XCTAssertLessThanOrEqual(
                    distribution[entry].probability, distribution[entry - 1].probability
                )
            }
            let sum = distribution.reduce(0.0) { $0 + $1.probability }
            XCTAssertEqual(sum, 1.0, accuracy: 1e-3)
        }
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
