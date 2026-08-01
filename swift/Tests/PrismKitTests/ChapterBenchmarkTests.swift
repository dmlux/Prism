import XCTest

@testable import PrismKit

/// Layer-by-layer wall-clock measurement of the complete pipeline on the
/// local book-chapter fixture. Run in release mode for meaningful numbers:
/// `swift test -c release --filter ChapterBenchmarkTests`
final class ChapterBenchmarkTests: XCTestCase {
    func testChapterEndToEndTiming() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let chapterURL = root.appendingPathComponent("data/examples/hp7kap1.txt")
        let artifactURL = root.appendingPathComponent("models/prism-no-0.2.0")
        for url in [chapterURL, artifactURL.appendingPathComponent("manifest.json")] {
            try XCTSkipUnless(
                FileManager.default.fileExists(atPath: url.path),
                "Local fixture is not present."
            )
        }
        let text = try String(contentsOf: chapterURL, encoding: .utf8)

        var stamp = Date()
        func lap(_ label: String) {
            let now = Date()
            print(String(format: "BENCH %@: %.1f ms", label, now.timeIntervalSince(stamp) * 1000))
            stamp = now
        }

        let tagger = try PrismTagger(artifactURL: artifactURL, device: .cpu)
        lap("load artifact + program + tokenizer")

        let policy = SegmentationPolicy.norwegian(maximumTokenCount: 96)
        let sentences = RuntimeSegmentation.segment(text, policy: policy)
        lap("segmentation (\(sentences.count) sentences)")

        let tokenizer = try SubwordTokenizer(
            vocabularyURL: artifactURL.appendingPathComponent("vocabulary.json")
        )
        lap("tokenizer init")
        var subwordTotal = 0
        for sentence in sentences {
            subwordTotal += tokenizer.encode(sentence).inputIds.count
        }
        lap("bpe (\(subwordTotal) subwords)")

        _ = try tagger.tag(sentences: [sentences[0]])
        lap("warmup batch")

        let started = Date()
        let tagged = try tagger.tag(sentences: sentences)
        let total = Date().timeIntervalSince(started)
        let tokenCount = tagged.reduce(0) { $0 + $1.tokens.count }
        print(String(
            format: "BENCH tag(sentences): %.0f ms for %d tokens = %.0f tokens/s",
            total * 1000, tokenCount, Double(tokenCount) / total
        ))
        XCTAssertEqual(tagged.count, sentences.count)
    }
}
