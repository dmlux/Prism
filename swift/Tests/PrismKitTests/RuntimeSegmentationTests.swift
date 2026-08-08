import Testing

@testable import PrismKit

/// Mirrors `python/tests/test_runtime_segmentation.py` so both language
/// implementations stay behaviourally identical.
struct RuntimeSegmentationTests {
    private let policy = SegmentationPolicy(
        abbreviationTokens: ["f.eks."],
        maximumTokenCount: 8
    )

    @Test func keepsFragmentsAndHeadings() {
        let text = "KAPITTEL 1\nHan gjekk heim.\nog so vidare"

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        #expect(
            sentences.map(\.tokens) == [
                ["KAPITTEL", "1"],
                ["Han", "gjekk", "heim", "."],
                ["og", "so", "vidare"],
            ]
        )
    }

    @Test func chunksLongSentencesWithoutLoss() {
        let words = (0..<19).map { "ord\($0)" }
        let text = words.joined(separator: " ") + "."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        #expect(sentences.count == 3)
        #expect(sentences.map(\.tokens.count) == [8, 8, 4])
        #expect(sentences.flatMap(\.tokens) == words + ["."])
        #expect(sentences.allSatisfy { $0.hasSpaceBefore.first == false })
    }

    @Test func restoresMissingSentenceSpaces() {
        let text = "De begynte å gå.De gikk fort.«Noe nytt?» spurte han om f.eks.Dette."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        #expect(
            sentences.map(\.tokens) == [
                ["De", "begynte", "å", "gå", "."],
                ["De", "gikk", "fort", "."],
                ["«", "Noe", "nytt", "?", "»", "spurte", "han", "om"],
                ["f.eks.", "Dette", "."],
            ]
        )
    }

    @Test func protectsAbbreviationsAndOrdinals() {
        let text = "Vi feirar 17. mai med f.eks. kake. Det er fint."

        let sentences = RuntimeSegmentation.segment(text, policy: policy)

        #expect(
            sentences.map(\.tokens) == [
                ["Vi", "feirar", "17.", "mai", "med", "f.eks.", "kake", "."],
                ["Det", "er", "fint", "."],
            ]
        )
    }

    @Test func mergesWrappedLinesWithDehyphenation() {
        let text = "Katten\nhennes sov.\nDen hadde vand-\nring i blodet."

        let sentences = RuntimeSegmentation.segment(
            text,
            policy: .norwegian(maximumTokenCount: 32)
        )

        #expect(
            sentences.map(\.tokens) == [
                ["Katten", "hennes", "sov", "."],
                ["Den", "hadde", "vandring", "i", "blodet", "."],
            ]
        )
    }

    @Test func spacingReflectsAttachedPunctuation() {
        let sentences = RuntimeSegmentation.segment(
            "Hun sa «nei», ikke sant?",
            policy: .norwegian()
        )

        #expect(sentences.count == 1)
        #expect(
            sentences[0].tokens
                == ["Hun", "sa", "«", "nei", "»", ",", "ikke", "sant", "?"]
        )
        #expect(
            sentences[0].hasSpaceBefore
                == [false, true, true, false, false, false, true, true, false]
        )
    }
}
