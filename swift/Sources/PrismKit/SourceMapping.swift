import Foundation

/// The canonical Prism source position: a half-open `[start, end)` byte
/// range in the **UTF-8 encoding** of the exact, unmodified string passed
/// to ``PrismTagger/tag(text:)``.
///
/// The offsets never refer to internally repaired, merged, or otherwise
/// transformed intermediate strings, and they are UTF-8 byte offsets — not
/// UTF-16 code units and not `Character` counts. Apple text APIs such as
/// TextKit and `NSRange` usually count UTF-16 code units: `"🙂å"` occupies
/// 4 + 2 UTF-8 bytes but 2 + 1 UTF-16 code units, so `å` starts at UTF-8
/// byte offset 4 yet at UTF-16 offset 2. Copying the numbers is therefore
/// not a conversion; convert against the original string, for example with
/// ``range(in:)``.
///
/// Every range Prism emits is non-empty, lies inside the input, has both
/// boundaries on UTF-8 codepoint boundaries, and range lists are ordered
/// and non-overlapping.
public struct Utf8ByteRange: Equatable, Hashable, Sendable {
    /// Inclusive UTF-8 byte offset.
    public let start: Int
    /// Exclusive UTF-8 byte offset.
    public let end: Int

    public init(start: Int, end: Int) {
        precondition(start >= 0 && start < end, "A Utf8ByteRange must satisfy 0 <= start < end.")
        self.start = start
        self.end = end
    }

    /// Maps this UTF-8 byte range onto a `String.Index` range of the exact
    /// original string.
    ///
    /// Returns `nil` when a boundary lies outside the string's UTF-8 view
    /// or does not fall on an index of the string — invalid boundaries are
    /// rejected instead of being rounded.
    public func range(in originalText: String) -> Range<String.Index>? {
        let utf8 = originalText.utf8
        guard
            let lowerUtf8 = utf8.index(
                utf8.startIndex, offsetBy: start, limitedBy: utf8.endIndex
            ),
            let upperUtf8 = utf8.index(
                utf8.startIndex, offsetBy: end, limitedBy: utf8.endIndex
            ),
            let lower = lowerUtf8.samePosition(in: originalText),
            let upper = upperUtf8.samePosition(in: originalText)
        else { return nil }
        return lower..<upper
    }
}
