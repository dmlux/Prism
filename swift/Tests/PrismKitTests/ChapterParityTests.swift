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
}
