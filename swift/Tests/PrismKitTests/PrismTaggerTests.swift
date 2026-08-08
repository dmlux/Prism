import Foundation
import Testing

@testable import PrismKit

/// End-to-end pipeline validation against decisions recorded from the
/// Python reference tagger on the same frozen artifact.
struct PrismTaggerTests {
    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static var artifactURL: URL {
        repositoryRoot.appendingPathComponent("models/prism-no-0.2.4")
    }

    static var fastArtifactURL: URL {
        repositoryRoot.appendingPathComponent("models/prism-no-0.2.4-fast")
    }

    private func loadTagger() throws -> PrismTagger {
        try PrismTagger(artifactURL: Self.artifactURL, device: .cpu)
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func tagsRawTextWithReferenceDecisions() throws {
        let tagger = try loadTagger()

        let sentences = try tagger.tag(text: "Hun kjøpte tre gamle bøker den 17. mai.")

        #expect(sentences.count == 1)
        let tokens = sentences[0].tokens
        #expect(
            tokens.map(\.text)
                == ["Hun", "kjøpte", "tre", "gamle", "bøker", "den", "17.", "mai", "."]
        )
        #expect(
            tokens.map(\.upos)
                == ["PRON", "VERB", "NUM", "ADJ", "NOUN", "DET", "ADJ", "NOUN", "PUNCT"]
        )
        #expect(
            tokens.map(\.lemma)
                == ["hun", "kjøpe", "tre", "gammel", "bok", "den", "17.", "mai", "."]
        )
        #expect(tokens[4].features["Gender"] == ["Fem"])
        #expect(tokens[4].features["Number"] == ["Plur"])
        #expect(tokens.allSatisfy { $0.uposConfidence > 0.9 })
        #expect(tokens.allSatisfy { $0.lemmaConfidence > 0.9 })
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func rawTextResultsCarrySourceRanges() throws {
        let tagger = try loadTagger()
        let text = "Hun kjøpte tre gamle bøker den 17. mai."

        let sentences = try tagger.tag(text: text)

        #expect(sentences.count == 1)
        // Byte offsets shared with the C++ and Java suites (parity).
        #expect(
            sentences[0].tokens.map(\.sourceRanges)
                == [
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
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 41)])
        let boker = sentences[0].tokens[4].sourceRanges[0].range(in: text)!
        #expect(String(text[boker]) == "bøker")
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func pretokenizedInputCarriesNoSourceRanges() throws {
        let tagger = try loadTagger()

        let sentences = try tagger.tag(pretokenized: [["Katten", "sov", "."]])

        #expect(sentences.count == 1)
        #expect(sentences[0].sourceRanges.isEmpty)
        #expect(sentences[0].tokens.allSatisfy { $0.sourceRanges.isEmpty })
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func callerProvidedSourceRangesPassThrough() throws {
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

        #expect(sentences.count == 1)
        #expect(sentences[0].sourceRanges == sentence.sourceRanges)
        #expect(sentences[0].tokens.map(\.sourceRanges) == sentence.tokenSourceRanges)
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func exposesArtifactMetadata() throws {
        let tagger = try loadTagger()

        #expect(tagger.artifactName == "prism-no")
        #expect(tagger.artifactVersion == "0.2.4")
        // Since 0.2.3 the manifest also declares the BCP 47 macrolanguage,
        // so plain-"no" documents match without host-side aliases.
        #expect(tagger.languageTags == ["nb", "nn", "no"])

        // Label inventories mirrored from labels.json.
        #expect(tagger.uposLabels.count == 17)
        #expect(tagger.uposLabels.contains("NOUN"))
        #expect(tagger.morphologyFeatures.count == 18)
        #expect(
            tagger.morphologyFeatures.first { $0.name == "Number" }?
                .values.contains("Plur") ?? false
        )
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func reportsTheUposDistributionPerToken() throws {
        let tagger = try loadTagger()

        let sentences = try tagger.tag(pretokenized: [["Katten", "sov", "."]])
        for token in sentences[0].tokens {
            let distribution = token.uposDistribution
            // One entry per artifact UPOS label, sorted by descending
            // probability; the first entry is the reported decision.
            #expect(distribution.count == tagger.uposLabels.count)
            #expect(distribution[0].upos == token.upos)
            #expect(distribution[0].probability == token.uposConfidence)
            for entry in 1..<distribution.count {
                #expect(distribution[entry].probability <= distribution[entry - 1].probability)
            }
            let sum = distribution.reduce(0.0) { $0 + $1.probability }
            #expect(abs(sum - 1.0) <= 1e-3)
        }
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func tagsMoreSentencesThanOneBatch() throws {
        let tagger = try loadTagger()
        let sentence = ["Katten", "sov", "."]

        let sentences = try tagger.tag(
            pretokenized: Array(repeating: sentence, count: 11)
        )

        #expect(sentences.count == 11)
        for tagged in sentences {
            #expect(tagged.tokens.map(\.upos) == ["NOUN", "VERB", "PUNCT"])
            #expect(tagged.tokens[0].lemma == "katt")
            #expect(tagged.tokens[1].lemma == "sove")
        }
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func batchSortingKeepsIdenticalSentencesAnchored() throws {
        let tagger = try loadTagger()
        // Twenty identical sentences force several batches and length-sorted
        // reordering; every result must still point at its own occurrence.
        let text = Array(repeating: "Katten sov.", count: 20).joined(separator: " ")

        let sentences = try tagger.tag(text: text)

        #expect(sentences.count == 20)
        for (index, sentence) in sentences.enumerated() {
            let base = index * 12
            #expect(sentence.sourceRanges == [Utf8ByteRange(start: base, end: base + 11)])
            #expect(
                sentence.tokens[0].sourceRanges[0]
                    == Utf8ByteRange(start: base, end: base + 6)
            )
        }
    }

    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.artifactURL),
        "Local artifact is not present."))
    func tagsExampleTextsEndToEnd() throws {
        // The checked-in CC0 example texts (see data/examples/README.md) with
        // the Python reference implementation's sentence and token counts,
        // through the full raw-text tagging pipeline.
        let tagger = try loadTagger()
        let repositoryRoot = Self.repositoryRoot
        let expectations: [(fixture: String, sentences: Int, tokens: Int)] = [
            ("skarvholmen-bokmaal", 55, 905),
            ("fjellvatnet-nynorsk", 41, 803),
        ]
        for expected in expectations {
            let text = try String(
                contentsOf: repositoryRoot.appendingPathComponent(
                    "data/examples/\(expected.fixture).txt"
                ),
                encoding: .utf8
            )
            let tagged = try tagger.tag(text: text)
            #expect(tagged.count == expected.sentences, "\(expected.fixture)")
            #expect(
                tagged.reduce(0) { $0 + $1.tokens.count } == expected.tokens,
                "\(expected.fixture)"
            )
        }
    }

    // The fast (int8) artifact must reproduce the same reference decisions;
    // quality is gated on the development split at export time, and this pins
    // the end-to-end runtime behaviour.
    @Test(.enabled(if: FixtureParity.artifactExists(PrismTaggerTests.fastArtifactURL),
        "Local fast artifact is not present."))
    func tagsFastRawTextWithReferenceDecisions() throws {
        let tagger = try PrismTagger(artifactURL: Self.fastArtifactURL, device: .cpu)

        let sentences = try tagger.tag(text: "Hun kjøpte tre gamle bøker den 17. mai.")

        #expect(sentences.count == 1)
        let tokens = sentences[0].tokens
        #expect(
            tokens.map(\.upos)
                == ["PRON", "VERB", "NUM", "ADJ", "NOUN", "DET", "ADJ", "NOUN", "PUNCT"]
        )
        #expect(
            tokens.map(\.lemma)
                == ["hun", "kjøpe", "tre", "gammel", "bok", "den", "17.", "mai", "."]
        )
        #expect(tokens.allSatisfy { $0.uposConfidence > 0.9 })
    }
}
