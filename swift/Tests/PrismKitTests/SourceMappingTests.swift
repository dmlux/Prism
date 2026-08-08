import XCTest

@testable import PrismKit

/// Source-mapping contract of the runtime segmentation. The expected byte
/// offsets are the same literals as in the C++ suite
/// (`cpp/tests/source_mapping_tests.cpp`), which pins byte-offset parity
/// across the bindings.
final class SourceMappingTests: XCTestCase {
    private let policy = SegmentationPolicy(
        abbreviationTokens: ["f.eks."],
        maximumTokenCount: 8
    )

    private func expectValidMapping(
        _ text: String,
        _ sentences: [PretokenizedSentence],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let byteCount = text.utf8.count
        for sentence in sentences {
            XCTAssertEqual(
                sentence.tokenSourceRanges.count, sentence.tokens.count,
                file: file, line: line
            )
            XCTAssertFalse(sentence.sourceRanges.isEmpty, file: file, line: line)
            for range in sentence.sourceRanges + sentence.tokenSourceRanges.flatMap({ $0 }) {
                XCTAssertLessThan(range.start, range.end, file: file, line: line)
                XCTAssertLessThanOrEqual(range.end, byteCount, file: file, line: line)
                // Both boundaries must map back onto the original string —
                // the helper rejects mid-codepoint offsets.
                XCTAssertNotNil(range.range(in: text), file: file, line: line)
            }
        }
    }

    func testAsciiTokensMapToExactByteRanges() {
        let text = "Katten sov. Hunden sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 2)
        expectValidMapping(text, sentences)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 6)],
                [Utf8ByteRange(start: 7, end: 10)],
                [Utf8ByteRange(start: 10, end: 11)],
            ]
        )
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 11)])
        XCTAssertEqual(sentences[1].sourceRanges, [Utf8ByteRange(start: 12, end: 23)])
    }

    func testNorwegianMultibyteLettersCountBytes() {
        let text = "Blåbær smaker godt."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].tokens[0], "Blåbær")
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 8)],
                [Utf8ByteRange(start: 9, end: 15)],
                [Utf8ByteRange(start: 16, end: 20)],
                [Utf8ByteRange(start: 20, end: 21)],
            ]
        )
    }

    func testEmojiBeforeAndBetweenTokens() {
        let before = "🙂 Katten sov."
        var sentences = RuntimeSegmentation.segment(before, policy: .norwegian())
        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(before, sentences)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 4)],
                [Utf8ByteRange(start: 5, end: 11)],
                [Utf8ByteRange(start: 12, end: 15)],
                [Utf8ByteRange(start: 15, end: 16)],
            ]
        )

        let between = "Katten 🙂 sov."
        sentences = RuntimeSegmentation.segment(between, policy: .norwegian())
        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(between, sentences)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges[1],
            [Utf8ByteRange(start: 7, end: 11)]
        )
    }

    func testDecomposedCombiningMarkStaysOnCodepointBoundaries() {
        // "a" plus combining ring (U+030A) is visually "å" but differently
        // encoded; every boundary stays a codepoint boundary of the input.
        let text = "a\u{030A} er fin."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 1)],
                [Utf8ByteRange(start: 1, end: 3)],
                [Utf8ByteRange(start: 4, end: 6)],
                [Utf8ByteRange(start: 7, end: 10)],
                [Utf8ByteRange(start: 10, end: 11)],
            ]
        )
    }

    func testRepeatedIdenticalTokensMapToDistinctOccurrences() {
        // A find()-style reconstruction would collapse onto the first
        // occurrence; the carried mapping assigns each repetition its bytes.
        let text = "ja ja ja ja."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(text, sentences)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 2)],
                [Utf8ByteRange(start: 3, end: 5)],
                [Utf8ByteRange(start: 6, end: 8)],
                [Utf8ByteRange(start: 9, end: 11)],
                [Utf8ByteRange(start: 11, end: 12)],
            ]
        )
    }

    func testRestoredSentenceSpaceKeepsOriginalOffsets() {
        let text = "Han går langs veien.Et sekund senere står han."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 2)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 21)])
        XCTAssertEqual(sentences[1].sourceRanges, [Utf8ByteRange(start: 21, end: 48)])
        XCTAssertEqual(sentences[1].tokens[0], "Et")
        XCTAssertEqual(
            sentences[1].tokenSourceRanges[0], [Utf8ByteRange(start: 21, end: 23)]
        )
    }

    func testDehyphenatedLineWrapKeepsBothFragments() {
        // The de-hyphenated model token stays "språkmodellen", but its
        // source mapping must point at the two contributing fragments —
        // never at a single invented range claiming "-\n" as token content.
        let text = "Dette er språk-\nmodellen til laget."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].tokens[2], "språkmodellen")
        XCTAssertEqual(
            sentences[0].tokenSourceRanges[2],
            [Utf8ByteRange(start: 9, end: 15), Utf8ByteRange(start: 17, end: 25)]
        )
        let fragments = sentences[0].tokenSourceRanges[2].map { fragment in
            String(text[fragment.range(in: text)!])
        }
        XCTAssertEqual(fragments, ["språk", "modellen"])
        for range in sentences[0].tokenSourceRanges.flatMap({ $0 }) {
            XCTAssertTrue(range.end <= 15 || range.start >= 17)
        }
        // The sentence splits at the removed hyphen instead of bridging it.
        XCTAssertEqual(
            sentences[0].sourceRanges,
            [Utf8ByteRange(start: 0, end: 15), Utf8ByteRange(start: 17, end: 36)]
        )
    }

    func testWrappedLineMergesAcrossNewlineWhitespace() {
        let text = "Katten\nhennes sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 18)])
    }

    func testCollapsedWhitespaceRunsKeepTokenOffsets() {
        let text = "Hun   sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 3)],
                [Utf8ByteRange(start: 6, end: 9)],
                [Utf8ByteRange(start: 9, end: 10)],
            ]
        )
    }

    func testAbbreviationUrlAndEmailStayContiguous() {
        var sentences = RuntimeSegmentation.segment("Vi har f.eks. kake.", policy: .norwegian())
        XCTAssertEqual(sentences[0].tokens[2], "f.eks.")
        XCTAssertEqual(
            sentences[0].tokenSourceRanges[2], [Utf8ByteRange(start: 7, end: 13)]
        )

        sentences = RuntimeSegmentation.segment("Se https://prism.no i dag.", policy: .norwegian())
        XCTAssertEqual(sentences[0].tokens[1], "https://prism.no")
        XCTAssertEqual(
            sentences[0].tokenSourceRanges[1], [Utf8ByteRange(start: 3, end: 19)]
        )

        sentences = RuntimeSegmentation.segment(
            "Skriv til post@prism.no i dag.", policy: .norwegian()
        )
        XCTAssertEqual(sentences[0].tokens[2], "post@prism.no")
        XCTAssertEqual(
            sentences[0].tokenSourceRanges[2], [Utf8ByteRange(start: 10, end: 23)]
        )
    }

    func testChunkingSlicesTokenRangesAndClipsSentenceRanges() {
        let words = (0..<19).map { "ord\($0)" }
        let text = words.joined(separator: " ") + "."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        XCTAssertEqual(sentences.count, 3)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[1].tokens[0], "ord8")
        XCTAssertEqual(
            sentences[1].tokenSourceRanges[0], [Utf8ByteRange(start: 40, end: 44)]
        )
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 39)])
        XCTAssertEqual(sentences[1].sourceRanges, [Utf8ByteRange(start: 40, end: 85)])
        XCTAssertEqual(sentences[2].sourceRanges, [Utf8ByteRange(start: 86, end: 104)])
    }

    func testRangeInStringMapsAndRejectsInvalidBounds() {
        let text = "🙂å ok"
        // å (2 bytes) follows the 4-byte emoji.
        let emoji = Utf8ByteRange(start: 0, end: 4)
        let aRing = Utf8ByteRange(start: 4, end: 6)
        XCTAssertEqual(String(text[emoji.range(in: text)!]), "🙂")
        XCTAssertEqual(String(text[aRing.range(in: text)!]), "å")

        // Mid-codepoint and out-of-bounds boundaries are rejected, not
        // rounded.
        XCTAssertNil(Utf8ByteRange(start: 1, end: 4).range(in: text))
        XCTAssertNil(Utf8ByteRange(start: 0, end: 5).range(in: text))
        XCTAssertNil(Utf8ByteRange(start: 0, end: 99).range(in: text))
    }

    func testLeadingAndTrailingWhitespaceShiftsRanges() {
        let text = "  Katten sov.  "
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(text, sentences)
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 2, end: 8)],
                [Utf8ByteRange(start: 9, end: 12)],
                [Utf8ByteRange(start: 12, end: 13)],
            ]
        )
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 2, end: 13)])
    }

    func testConsecutiveMultibyteLettersFormOneToken() {
        let text = "æøå er bokstaver."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 1)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].tokens[0], "æøå")
        XCTAssertEqual(
            sentences[0].tokenSourceRanges,
            [
                [Utf8ByteRange(start: 0, end: 6)],
                [Utf8ByteRange(start: 7, end: 9)],
                [Utf8ByteRange(start: 10, end: 19)],
                [Utf8ByteRange(start: 19, end: 20)],
            ]
        )
    }

    func testRepeatedIdenticalSentencesMapToDistinctOccurrences() {
        let text = "Han sov. Han sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 2)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 8)])
        XCTAssertEqual(sentences[1].sourceRanges, [Utf8ByteRange(start: 9, end: 17)])
        XCTAssertEqual(sentences[0].tokenSourceRanges[0], [Utf8ByteRange(start: 0, end: 3)])
        XCTAssertEqual(sentences[1].tokenSourceRanges[0], [Utf8ByteRange(start: 9, end: 12)])
    }

    func testMultipleRepairedBoundariesStayAligned() {
        let text = "De gikk.De kom.De sov."
        let sentences = RuntimeSegmentation.segment(text, policy: .norwegian())

        XCTAssertEqual(sentences.count, 3)
        expectValidMapping(text, sentences)
        XCTAssertEqual(sentences[0].sourceRanges, [Utf8ByteRange(start: 0, end: 8)])
        XCTAssertEqual(sentences[1].sourceRanges, [Utf8ByteRange(start: 8, end: 15)])
        XCTAssertEqual(sentences[2].sourceRanges, [Utf8ByteRange(start: 15, end: 22)])
    }

    func testChunkHelperSlicesCallerProvidedRanges() {
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

        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(
            chunks[0].tokenSourceRanges,
            [[Utf8ByteRange(start: 0, end: 1)], [Utf8ByteRange(start: 2, end: 3)]]
        )
        XCTAssertEqual(chunks[0].sourceRanges, [Utf8ByteRange(start: 0, end: 3)])
        XCTAssertEqual(chunks[1].tokenSourceRanges, [[Utf8ByteRange(start: 4, end: 5)]])
        XCTAssertEqual(chunks[1].sourceRanges, [Utf8ByteRange(start: 4, end: 5)])
    }

    func testEveryFixtureTokenStaysAnchored() throws {
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
                let start = try XCTUnwrap(sentence.sourceRanges.first).start
                XCTAssertGreaterThanOrEqual(start, previousStart, fixture)
                previousStart = start
            }
        }
    }
}
