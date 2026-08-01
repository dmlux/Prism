import XCTest

@testable import PrismKit

/// Mirrors `python/tests/test_runtime_segmentation.py` so both language
/// implementations stay behaviourally identical.
final class RuntimeSegmentationTests: XCTestCase {
    private let policy = SegmentationPolicy(
        abbreviationTokens: ["f.eks."],
        maximumTokenCount: 8
    )

    func testKeepsFragmentsAndHeadings() {
        let text = "KAPITTEL 1\nHan gjekk heim.\nog so vidare"

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        XCTAssertEqual(
            sentences.map(\.tokens),
            [
                ["KAPITTEL", "1"],
                ["Han", "gjekk", "heim", "."],
                ["og", "so", "vidare"],
            ]
        )
    }

    func testChunksLongSentencesWithoutLoss() {
        let words = (0..<19).map { "ord\($0)" }
        let text = words.joined(separator: " ") + "."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        XCTAssertEqual(sentences.count, 3)
        XCTAssertEqual(sentences.map(\.tokens.count), [8, 8, 4])
        XCTAssertEqual(
            sentences.flatMap(\.tokens),
            words + ["."]
        )
        XCTAssertTrue(sentences.allSatisfy { $0.hasSpaceBefore.first == false })
    }

    func testRestoresMissingSentenceSpaces() {
        let text = "De begynte å gå.De gikk fort.«Noe nytt?» spurte han om f.eks.Dette."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        XCTAssertEqual(
            sentences.map(\.tokens),
            [
                ["De", "begynte", "å", "gå", "."],
                ["De", "gikk", "fort", "."],
                ["«", "Noe", "nytt", "?", "»", "spurte", "han", "om"],
                ["f.eks.", "Dette", "."],
            ]
        )
    }

    func testProtectsAbbreviationsAndOrdinals() {
        let text = "Vi feirar 17. mai med f.eks. kake. Det er fint."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        XCTAssertEqual(
            sentences.map(\.tokens),
            [
                ["Vi", "feirar", "17.", "mai", "med", "f.eks.", "kake", "."],
                ["Det", "er", "fint", "."],
            ]
        )
    }

    func testMergesWrappedLinesWithDehyphenation() {
        let text = "Katten\nhennes sov.\nDen hadde vand-\nring i blodet."

        let sentences = RuntimeSegmentation.segment(
            text,
            policy: .norwegian(maximumTokenCount: 32)
        )

        XCTAssertEqual(
            sentences.map(\.tokens),
            [
                ["Katten", "hennes", "sov", "."],
                ["Den", "hadde", "vandring", "i", "blodet", "."],
            ]
        )
    }

    func testSpacingReflectsAttachedPunctuation() {
        let sentences = RuntimeSegmentation.segment(
            "Hun sa «nei», ikke sant?",
            policy: .norwegian()
        )

        XCTAssertEqual(sentences.count, 1)
        XCTAssertEqual(
            sentences[0].tokens,
            ["Hun", "sa", "«", "nei", "»", ",", "ikke", "sant", "?"]
        )
        XCTAssertEqual(
            sentences[0].hasSpaceBefore,
            [false, true, true, false, false, false, true, true, false]
        )
    }
}
