import Foundation
import Testing

@testable import PrismKit

/// Source-mapping contract of the runtime segmentation. The expected byte
/// offsets are the same literals as in the C++ suite
/// (`cpp/tests/source_mapping_tests.cpp`), which pins byte-offset parity
/// across the bindings.
struct SourceMappingTests {
    private let policy = SegmentationPolicy(
        abbreviationTokens: ["f.eks."],
        maximumTokenCount: 8
    )

    private func expectValidMapping(
        _ text: String,
        _ sentences: [PretokenizedSentence],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let byteCount = text.utf8.count
        for sentence in sentences {
            #expect(
                sentence.tokenSourceRanges.count == sentence.tokens.count,
                sourceLocation: sourceLocation
            )
            #expect(!sentence.sourceRanges.isEmpty, sourceLocation: sourceLocation)
            for range in sentence.sourceRanges + sentence.tokenSourceRanges.flatMap({ $0 }) {
                #expect(range.start < range.end, sourceLocation: sourceLocation)
                #expect(range.end <= byteCount, sourceLocation: sourceLocation)
                // Both boundaries must map back onto the original string —
                // the helper rejects mid-codepoint offsets.
                #expect(range.range(in: text) != nil, sourceLocation: sourceLocation)
            }
        }
    }

    @Test func asciiTokensMapToExactByteRanges() {
        let text = "Katten sov. Hunden sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 2)
        expectValidMapping(text, sentences)
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 6)],
                    [Utf8ByteRange(start: 7, end: 10)],
                    [Utf8ByteRange(start: 10, end: 11)],
                ]
        )
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 11)])
        #expect(sentences[1].sourceRanges == [Utf8ByteRange(start: 12, end: 23)])
    }

    @Test func norwegianMultibyteLettersCountBytes() {
        let text = "Blåbær smaker godt."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        expectValidMapping(text, sentences)
        #expect(sentences[0].tokens[0] == "Blåbær")
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 8)],
                    [Utf8ByteRange(start: 9, end: 15)],
                    [Utf8ByteRange(start: 16, end: 20)],
                    [Utf8ByteRange(start: 20, end: 21)],
                ]
        )
    }

    @Test func emojiBeforeAndBetweenTokens() {
        let before = "🙂 Katten sov."
        var sentences = RuntimeSegmentation.segment(before, policy: .norwegian())
        #expect(sentences.count == 1)
        expectValidMapping(before, sentences)
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 4)],
                    [Utf8ByteRange(start: 5, end: 11)],
                    [Utf8ByteRange(start: 12, end: 15)],
                    [Utf8ByteRange(start: 15, end: 16)],
                ]
        )

        let between = "Katten 🙂 sov."
        sentences = RuntimeSegmentation.segment(between, policy: .norwegian())
        #expect(sentences.count == 1)
        expectValidMapping(between, sentences)
        #expect(
            sentences[0].tokenSourceRanges[1]
                == [Utf8ByteRange(start: 7, end: 11)]
        )
    }

    @Test func decomposedCombiningMarkStaysOnCodepointBoundaries() {
        // "a" plus combining ring (U+030A) is visually "å" but differently
        // encoded; every boundary stays a codepoint boundary of the input.
        let text = "a\u{030A} er fin."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 1)],
                    [Utf8ByteRange(start: 1, end: 3)],
                    [Utf8ByteRange(start: 4, end: 6)],
                    [Utf8ByteRange(start: 7, end: 10)],
                    [Utf8ByteRange(start: 10, end: 11)],
                ]
        )
    }

    @Test func repeatedIdenticalTokensMapToDistinctOccurrences() {
        // A find()-style reconstruction would collapse onto the first
        // occurrence; the carried mapping assigns each repetition its bytes.
        let text = "ja ja ja ja."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        expectValidMapping(text, sentences)
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 2)],
                    [Utf8ByteRange(start: 3, end: 5)],
                    [Utf8ByteRange(start: 6, end: 8)],
                    [Utf8ByteRange(start: 9, end: 11)],
                    [Utf8ByteRange(start: 11, end: 12)],
                ]
        )
    }

    @Test func restoredSentenceSpaceKeepsOriginalOffsets() {
        let text = "Han går langs veien.Et sekund senere står han."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 2)
        expectValidMapping(text, sentences)
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 21)])
        #expect(sentences[1].sourceRanges == [Utf8ByteRange(start: 21, end: 48)])
        #expect(sentences[1].tokens[0] == "Et")
        #expect(sentences[1].tokenSourceRanges[0] == [Utf8ByteRange(start: 21, end: 23)])
    }

    @Test func dehyphenatedLineWrapKeepsBothFragments() {
        // The de-hyphenated model token stays "språkmodellen", but its
        // source mapping must point at the two contributing fragments —
        // never at a single invented range claiming "-\n" as token content.
        let text = "Dette er språk-\nmodellen til laget."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        expectValidMapping(text, sentences)
        #expect(sentences[0].tokens[2] == "språkmodellen")
        #expect(
            sentences[0].tokenSourceRanges[2]
                == [Utf8ByteRange(start: 9, end: 15), Utf8ByteRange(start: 17, end: 25)]
        )
        let fragments = sentences[0].tokenSourceRanges[2].map { fragment in
            String(text[fragment.range(in: text)!])
        }
        #expect(fragments == ["språk", "modellen"])
        for range in sentences[0].tokenSourceRanges.flatMap({ $0 }) {
            #expect(range.end <= 15 || range.start >= 17)
        }
        // The sentence splits at the removed hyphen instead of bridging it.
        #expect(
            sentences[0].sourceRanges
                == [Utf8ByteRange(start: 0, end: 15), Utf8ByteRange(start: 17, end: 36)]
        )
    }

    @Test func wrappedLineMergesAcrossNewlineWhitespace() {
        let text = "Katten\nhennes sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        expectValidMapping(text, sentences)
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 18)])
    }

    @Test func collapsedWhitespaceRunsKeepTokenOffsets() {
        let text = "Hun   sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 3)],
                    [Utf8ByteRange(start: 6, end: 9)],
                    [Utf8ByteRange(start: 9, end: 10)],
                ]
        )
    }

    @Test func abbreviationUrlAndEmailStayContiguous() {
        var sentences = RuntimeSegmentation.segment("Vi har f.eks. kake.", policy: .norwegian())
        #expect(sentences[0].tokens[2] == "f.eks.")
        #expect(sentences[0].tokenSourceRanges[2] == [Utf8ByteRange(start: 7, end: 13)])

        sentences = RuntimeSegmentation.segment("Se https://prism.no i dag.", policy: .norwegian())
        #expect(sentences[0].tokens[1] == "https://prism.no")
        #expect(sentences[0].tokenSourceRanges[1] == [Utf8ByteRange(start: 3, end: 19)])

        sentences = RuntimeSegmentation.segment(
            "Skriv til post@prism.no i dag.", policy: .norwegian()
        )
        #expect(sentences[0].tokens[2] == "post@prism.no")
        #expect(sentences[0].tokenSourceRanges[2] == [Utf8ByteRange(start: 10, end: 23)])
    }

    @Test func chunkingSlicesTokenRangesAndClipsSentenceRanges() {
        let words = (0..<19).map { "ord\($0)" }
        let text = words.joined(separator: " ") + "."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        #expect(sentences.count == 3)
        expectValidMapping(text, sentences)
        #expect(sentences[1].tokens[0] == "ord8")
        #expect(sentences[1].tokenSourceRanges[0] == [Utf8ByteRange(start: 40, end: 44)])
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 39)])
        #expect(sentences[1].sourceRanges == [Utf8ByteRange(start: 40, end: 85)])
        #expect(sentences[2].sourceRanges == [Utf8ByteRange(start: 86, end: 104)])
    }

    @Test func rangeInStringMapsAndRejectsInvalidBounds() {
        let text = "🙂å ok"
        // å (2 bytes) follows the 4-byte emoji.
        let emoji = Utf8ByteRange(start: 0, end: 4)
        let aRing = Utf8ByteRange(start: 4, end: 6)
        #expect(String(text[emoji.range(in: text)!]) == "🙂")
        #expect(String(text[aRing.range(in: text)!]) == "å")

        // Mid-codepoint and out-of-bounds boundaries are rejected, not
        // rounded.
        #expect(Utf8ByteRange(start: 1, end: 4).range(in: text) == nil)
        #expect(Utf8ByteRange(start: 0, end: 5).range(in: text) == nil)
        #expect(Utf8ByteRange(start: 0, end: 99).range(in: text) == nil)
    }

    @Test func leadingAndTrailingWhitespaceShiftsRanges() {
        let text = "  Katten sov.  "
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        expectValidMapping(text, sentences)
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 2, end: 8)],
                    [Utf8ByteRange(start: 9, end: 12)],
                    [Utf8ByteRange(start: 12, end: 13)],
                ]
        )
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 2, end: 13)])
    }

    @Test func consecutiveMultibyteLettersFormOneToken() {
        let text = "æøå er bokstaver."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 1)
        expectValidMapping(text, sentences)
        #expect(sentences[0].tokens[0] == "æøå")
        #expect(
            sentences[0].tokenSourceRanges
                == [
                    [Utf8ByteRange(start: 0, end: 6)],
                    [Utf8ByteRange(start: 7, end: 9)],
                    [Utf8ByteRange(start: 10, end: 19)],
                    [Utf8ByteRange(start: 19, end: 20)],
                ]
        )
    }

    @Test func repeatedIdenticalSentencesMapToDistinctOccurrences() {
        let text = "Han sov. Han sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 2)
        expectValidMapping(text, sentences)
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 8)])
        #expect(sentences[1].sourceRanges == [Utf8ByteRange(start: 9, end: 17)])
        #expect(sentences[0].tokenSourceRanges[0] == [Utf8ByteRange(start: 0, end: 3)])
        #expect(sentences[1].tokenSourceRanges[0] == [Utf8ByteRange(start: 9, end: 12)])
    }

    @Test func multipleRepairedBoundariesStayAligned() {
        let text = "De gikk.De kom.De sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        #expect(sentences.count == 3)
        expectValidMapping(text, sentences)
        #expect(sentences[0].sourceRanges == [Utf8ByteRange(start: 0, end: 8)])
        #expect(sentences[1].sourceRanges == [Utf8ByteRange(start: 8, end: 15)])
        #expect(sentences[2].sourceRanges == [Utf8ByteRange(start: 15, end: 22)])
    }

    @Test func chunkHelperSlicesCallerProvidedRanges() {
        let sentence = PretokenizedSentence(
            tokens: ["a", "b", "c"],
            hasSpaceBefore: [false, true, true],
            tokenSourceRanges: [
                [Utf8ByteRange(start: 0, end: 1)],
                [Utf8ByteRange(start: 2, end: 3)],
                [Utf8ByteRange(start: 4, end: 5)],
            ],
            sourceRanges: [Utf8ByteRange(start: 0, end: 5)]
        )

        let chunks = RuntimeSegmentation.chunk(sentence, maximumTokenCount: 2)

        #expect(chunks.count == 2)
        #expect(
            chunks[0].tokenSourceRanges
                == [[Utf8ByteRange(start: 0, end: 1)], [Utf8ByteRange(start: 2, end: 3)]]
        )
        #expect(chunks[0].sourceRanges == [Utf8ByteRange(start: 0, end: 3)])
        #expect(chunks[1].tokenSourceRanges == [[Utf8ByteRange(start: 4, end: 5)]])
        #expect(chunks[1].sourceRanges == [Utf8ByteRange(start: 4, end: 5)])
    }

    @Test func everyFixtureTokenStaysAnchored() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        for fixture in ["skarvholmen-bokmaal", "fjellvatnet-nynorsk"] {
            let textURL = repositoryRoot.appendingPathComponent(
                "data/examples/\(fixture).txt"
            )
            let text = try String(contentsOf: textURL, encoding: .utf8)

            let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())
            expectValidMapping(text, sentences)

            // Document order: sentence ranges never move backwards.
            var previousStart = 0
            for sentence in sentences {
                let start = try #require(sentence.sourceRanges.first).start
                #expect(start >= previousStart, "\(fixture)")
                previousStart = start
            }
        }
    }
}
