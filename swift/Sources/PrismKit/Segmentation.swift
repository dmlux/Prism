import Foundation

/// One sentence as UD-convention word tokens plus spacing information.
///
/// `tokenSourceRanges` and `sourceRanges` carry the source mapping against
/// the exact raw-text input (see ``Utf8ByteRange`` for the offset
/// contract). ``RuntimeSegmentation/segment(_:policy:)`` always fills both;
/// sentences assembled from bare tokens leave them empty, which
/// unambiguously means "no source positions available". Callers who own
/// their tokenization and offsets may fill the fields themselves; the
/// invariants (matching counts, non-empty, ordered, non-overlapping) are
/// enforced, while codepoint alignment stays the caller's contract.
public struct PretokenizedSentence: Equatable, Sendable {
    public let tokens: [String]
    public let hasSpaceBefore: [Bool]
    /// Per token, the ordered non-overlapping fragments of the original
    /// text the token was built from; a de-hyphenated line wrap yields one
    /// fragment per contributing piece.
    public let tokenSourceRanges: [[Utf8ByteRange]]
    /// Sentence cover: token fragments whose gap in the original is pure
    /// whitespace share one range; gaps containing removed non-whitespace
    /// content (for example a line-break hyphen) split the sentence.
    public let sourceRanges: [Utf8ByteRange]

    public init(
        tokens: [String],
        hasSpaceBefore: [Bool],
        tokenSourceRanges: [[Utf8ByteRange]] = [],
        sourceRanges: [Utf8ByteRange] = []
    ) {
        precondition(tokens.count == hasSpaceBefore.count)
        precondition(
            tokenSourceRanges.isEmpty || tokenSourceRanges.count == tokens.count,
            "tokenSourceRanges must be empty or hold exactly one range list per token."
        )
        // Ordering and non-overlap hold across the whole sentence, so
        // repeated identical tokens stay bound to their own occurrences.
        precondition(
            !tokenSourceRanges.contains(where: \.isEmpty)
                && Self.isOrderedNonOverlapping(tokenSourceRanges.flatMap { $0 }),
            "Every token needs at least one Utf8ByteRange, ordered and "
                + "non-overlapping across the sentence."
        )
        precondition(
            Self.isOrderedNonOverlapping(sourceRanges),
            "Sentence source ranges must be ordered and non-overlapping."
        )
        self.tokens = tokens
        self.hasSpaceBefore = hasSpaceBefore
        self.tokenSourceRanges = tokenSourceRanges
        self.sourceRanges = sourceRanges
    }

    static func isOrderedNonOverlapping(_ ranges: [Utf8ByteRange]) -> Bool {
        var previousEnd = 0
        for range in ranges {
            if range.start < previousEnd { return false }
            previousEnd = range.end
        }
        return true
    }
}

/// Versioned, language-configurable runtime segmentation policy.
///
/// This is the native port of Prism's `prism-runtime-segmentation-v1`: the
/// recall-oriented raw-text front end that turns application text into
/// UD-convention word tokens without ever dropping user content. Headings and
/// fragments become sentences, sentences beyond ``maximumTokenCount`` are
/// chunked into windows, and spaces lost after sentence punctuation (a common
/// e-book extraction defect) are restored deterministically.
public struct SegmentationPolicy: Sendable {
    public static let version = "prism-runtime-segmentation-v1"

    /// Lowercase abbreviations including their trailing period ("f.eks.").
    public let abbreviationTokens: Set<String>
    public let maximumTokenCount: Int

    public init(abbreviationTokens: Set<String>, maximumTokenCount: Int) {
        precondition(maximumTokenCount > 0)
        self.abbreviationTokens = abbreviationTokens
        self.maximumTokenCount = maximumTokenCount
    }

    /// The Norwegian policy matching the Python reference implementation.
    public static func norwegian(maximumTokenCount: Int = 128) -> SegmentationPolicy {
        SegmentationPolicy(
            abbreviationTokens: norwegianAbbreviations,
            maximumTokenCount: maximumTokenCount
        )
    }
}

let norwegianAbbreviations: Set<String> = [
    "adm.", "ang.", "bl.a.", "ca.", "d.v.s.", "dr.", "dvs.", "eks.",
    "ekskl.", "evt.", "f.eks.", "f.o.m.", "fylkeskomm.", "hhv.", "ifm.",
    "iht.", "inkl.", "jf.", "jfr.", "kap.", "kfr.", "kl.", "kr.", "m.a.",
    "m.fl.", "m.m.", "m.v.", "mht.", "mill.", "mrd.", "mv.", "nr.", "osv.",
    "p.g.a.", "pga.", "pkt.", "ref.", "saksnr.", "st.", "t.o.m.", "tlf.",
    "vedr.",
]

/// Raw-text segmentation into UD-convention pretokenized sentences.
///
/// A hand-written UTF-8 codepoint scanner sharing one semantics with the
/// C++ implementation, so segmentation behaviour *and* the emitted UTF-8
/// byte offsets are identical across the bindings. Every codepoint of the
/// transformed text travels with its origin in the raw input; internal
/// repairs (restored spaces, merged line wraps, collapsed whitespace,
/// removed line-break hyphens) therefore never lose the relationship to
/// the original text, and source ranges are never reconstructed by
/// searching the input afterwards.
public enum RuntimeSegmentation {
    // One codepoint of (possibly transformed) text together with its origin
    // in the raw input. Synthesized characters — the space restored after
    // sentence punctuation and the joining space of merged lines — carry an
    // empty range positioned at the insertion point. Tokens never contain
    // whitespace, so every codepoint inside a token has a real origin.
    struct TrackedCodepoint {
        var value: UInt32
        var sourceStart: Int
        var sourceEnd: Int

        var isSynthesized: Bool { sourceStart == sourceEnd }
    }

    /// Segment raw text without discarding content.
    public static func segment(
        _ text: String,
        policy: SegmentationPolicy
    ) -> [PretokenizedSentence] {
        let bytes = Array(text.utf8)
        let repaired = restoreMissingSentenceSpaces(decodeUtf8(bytes))
        var sentences: [PretokenizedSentence] = []
        for paragraph in mergeWrappedLines(repaired) {
            for sentenceCodepoints in splitParagraphSentences(paragraph, policy: policy) {
                appendChunks(
                    to: &sentences,
                    sentence: tokenizeWithSpacing(
                        sentenceCodepoints, policy: policy, originalBytes: bytes
                    ),
                    maximumTokenCount: policy.maximumTokenCount
                )
            }
        }
        return sentences
    }

    /// Split an over-long sentence into chunks of at most `maximumTokenCount`
    /// tokens. Source mappings, when present, travel with their tokens; each
    /// chunk's sentence ranges are the parent's ranges clipped to the chunk's
    /// own token fragments.
    static func chunk(
        _ sentence: PretokenizedSentence,
        maximumTokenCount: Int
    ) -> [PretokenizedSentence] {
        var chunks: [PretokenizedSentence] = []
        appendChunks(to: &chunks, sentence: sentence, maximumTokenCount: maximumTokenCount)
        return chunks
    }

    // MARK: - UTF-8 decoding

    static func decodeUtf8(_ bytes: [UInt8]) -> [TrackedCodepoint] {
        var result: [TrackedCodepoint] = []
        result.reserveCapacity(bytes.count)

        func isContinuation(_ byte: UInt8) -> Bool { byte & 0xC0 == 0x80 }

        var index = 0
        while index < bytes.count {
            let byte = bytes[index]
            var value: UInt32 = 0
            var byteCount = 1

            if byte < 0x80 {
                value = UInt32(byte)
            } else if byte & 0xE0 == 0xC0, index + 1 < bytes.count,
                isContinuation(bytes[index + 1])
            {
                value = (UInt32(byte & 0x1F) << 6) | UInt32(bytes[index + 1] & 0x3F)
                byteCount = 2
            } else if byte & 0xF0 == 0xE0, index + 2 < bytes.count,
                isContinuation(bytes[index + 1]), isContinuation(bytes[index + 2])
            {
                value = (UInt32(byte & 0x0F) << 12)
                    | (UInt32(bytes[index + 1] & 0x3F) << 6)
                    | UInt32(bytes[index + 2] & 0x3F)
                byteCount = 3
            } else if byte & 0xF8 == 0xF0, index + 3 < bytes.count,
                isContinuation(bytes[index + 1]), isContinuation(bytes[index + 2]),
                isContinuation(bytes[index + 3])
            {
                value = (UInt32(byte & 0x07) << 18)
                    | (UInt32(bytes[index + 1] & 0x3F) << 12)
                    | (UInt32(bytes[index + 2] & 0x3F) << 6)
                    | UInt32(bytes[index + 3] & 0x3F)
                byteCount = 4
            } else {
                value = 0xFFFD
            }

            result.append(
                TrackedCodepoint(
                    value: value,
                    sourceStart: index,
                    sourceEnd: min(index + byteCount, bytes.count)
                )
            )
            index += byteCount
        }
        return result
    }

    static func appendUtf8(to bytes: inout [UInt8], value: UInt32) {
        if value <= 0x7F {
            bytes.append(UInt8(value))
        } else if value <= 0x7FF {
            bytes.append(UInt8(0xC0 | ((value >> 6) & 0x1F)))
            bytes.append(UInt8(0x80 | (value & 0x3F)))
        } else if value <= 0xFFFF {
            bytes.append(UInt8(0xE0 | ((value >> 12) & 0x0F)))
            bytes.append(UInt8(0x80 | ((value >> 6) & 0x3F)))
            bytes.append(UInt8(0x80 | (value & 0x3F)))
        } else {
            bytes.append(UInt8(0xF0 | ((value >> 18) & 0x07)))
            bytes.append(UInt8(0x80 | ((value >> 12) & 0x3F)))
            bytes.append(UInt8(0x80 | ((value >> 6) & 0x3F)))
            bytes.append(UInt8(0x80 | (value & 0x3F)))
        }
    }

    static func text(
        of codepoints: [TrackedCodepoint], from start: Int, to end: Int
    ) -> String {
        var bytes: [UInt8] = []
        for index in start..<end {
            appendUtf8(to: &bytes, value: codepoints[index].value)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Character classes (shared semantics with the C++ scanner)

    static func isWhitespace(_ value: UInt32) -> Bool {
        switch value {
        case 0x20, 0x09, 0x0A, 0x0D, 0x0C, 0x0B,
            0x00A0, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
            return true
        default:
            return value >= 0x2000 && value <= 0x200A
        }
    }

    static func isDigit(_ value: UInt32) -> Bool {
        value >= 0x30 && value <= 0x39
    }

    static func isLetter(_ value: UInt32) -> Bool {
        if (value >= 0x61 && value <= 0x7A) || (value >= 0x41 && value <= 0x5A) {
            return true
        }
        if value < 0xC0 { return false }
        return (value <= 0x24F && value != 0xD7 && value != 0xF7)
            || (value >= 0x0370 && value <= 0x03FF && value != 0x03A2)
            || (value >= 0x0400 && value <= 0x052F)
    }

    // Mirrors the reference implementation's \w (letters, digits, underscore).
    static func isWordCharacter(_ value: UInt32) -> Bool {
        isLetter(value) || isDigit(value) || value == 0x5F
    }

    static func isLowercaseLetter(_ value: UInt32) -> Bool {
        if value >= 0x61 && value <= 0x7A { return true }
        return (value >= 0xDF && value <= 0xFF && value != 0xF7)
            || (value >= 0x100 && value <= 0x17F && value % 2 == 1)
            || (value >= 0x3B1 && value <= 0x3C9)
            || (value >= 0x430 && value <= 0x44F)
    }

    static func isUppercaseLetter(_ value: UInt32) -> Bool {
        if value >= 0x41 && value <= 0x5A { return true }
        return (value >= 0xC0 && value <= 0xDE && value != 0xD7)
            || (value >= 0x100 && value <= 0x17F && value % 2 == 0)
            || (value >= 0x391 && value <= 0x3A9 && value != 0x3A2)
            || (value >= 0x410 && value <= 0x42F)
    }

    static func isTerminal(_ value: UInt32) -> Bool {
        value == 0x2E || value == 0x21 || value == 0x3F || value == 0x2026
    }

    static func isOpening(_ value: UInt32) -> Bool {
        switch value {
        case 0x00AB, 0x22, 0x27, 0x28, 0x5B, 0x201E, 0x201C:
            return true
        default:
            return false
        }
    }

    static func lowercase(
        _ codepoints: [TrackedCodepoint], from start: Int, to end: Int
    ) -> String {
        var bytes: [UInt8] = []
        for index in start..<end {
            var value = codepoints[index].value
            if value >= 0x41 && value <= 0x5A {
                value += 32
            } else if value >= 0xC0 && value <= 0xDE && value != 0xD7 {
                value += 32
            }
            appendUtf8(to: &bytes, value: value)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    static func lowercase(_ text: String) -> String {
        let codepoints = decodeUtf8(Array(text.utf8))
        return lowercase(codepoints, from: 0, to: codepoints.count)
    }

    // MARK: - Repairs and paragraph assembly

    /// Restore spaces lost after sentence punctuation ("veien.Et sekund").
    /// The restored space is synthesized: it owns no bytes of the input.
    static func restoreMissingSentenceSpaces(
        _ codepoints: [TrackedCodepoint]
    ) -> [TrackedCodepoint] {
        var repaired: [TrackedCodepoint] = []
        repaired.reserveCapacity(codepoints.count + 16)

        for index in codepoints.indices {
            let current = codepoints[index]
            repaired.append(current)
            if index == 0 || index + 1 >= codepoints.count { continue }
            if isTerminal(current.value),
                isLowercaseLetter(codepoints[index - 1].value),
                isUppercaseLetter(codepoints[index + 1].value)
                    || isOpening(codepoints[index + 1].value)
            {
                repaired.append(
                    TrackedCodepoint(
                        value: 0x20,
                        sourceStart: current.sourceEnd,
                        sourceEnd: current.sourceEnd
                    )
                )
            }
        }
        return repaired
    }

    /// Join wrapped lines into paragraphs without joining layout lines.
    /// Joining spaces are synthesized; a popped line-break hyphen drops out
    /// of the mapping, so a de-hyphenated token keeps one fragment per line.
    static func mergeWrappedLines(
        _ text: [TrackedCodepoint]
    ) -> [[TrackedCodepoint]] {
        var paragraphs: [[TrackedCodepoint]] = []
        var paragraph: [TrackedCodepoint] = []

        func flush() {
            if !paragraph.isEmpty { paragraphs.append(paragraph) }
            paragraph = []
        }

        var lineStart = 0
        while lineStart <= text.count {
            var lineEnd = lineStart
            while lineEnd < text.count, text[lineEnd].value != 0x0A {
                lineEnd += 1
            }

            // Collapse internal whitespace runs like " ".join(line.split()).
            var line: [TrackedCodepoint] = []
            var previousSpace = true
            for index in lineStart..<lineEnd {
                let codepoint = text[index]
                if isWhitespace(codepoint.value) {
                    previousSpace = true
                    continue
                }
                if previousSpace, !line.isEmpty {
                    line.append(
                        TrackedCodepoint(
                            value: 0x20,
                            sourceStart: codepoint.sourceStart,
                            sourceEnd: codepoint.sourceStart
                        )
                    )
                }
                previousSpace = false
                line.append(codepoint)
            }

            if line.isEmpty {
                flush()
            } else {
                let continuesPrevious = !paragraph.isEmpty
                    && !isTerminal(paragraph.last!.value)
                    && paragraph.last!.value != 0x3A
                    && isLowercaseLetter(line.first!.value)
                if continuesPrevious, paragraph.last!.value == 0x2D {
                    paragraph.removeLast()
                    paragraph.append(contentsOf: line)
                } else if continuesPrevious {
                    paragraph.append(
                        TrackedCodepoint(
                            value: 0x20,
                            sourceStart: line.first!.sourceStart,
                            sourceEnd: line.first!.sourceStart
                        )
                    )
                    paragraph.append(contentsOf: line)
                } else {
                    flush()
                    paragraph = line
                }
            }

            if lineEnd >= text.count { break }
            lineStart = lineEnd + 1
        }
        flush()
        return paragraphs
    }

    static func isProtectedBoundary(
        _ codepoints: [TrackedCodepoint],
        matchStart: Int,
        matchEnd: Int,
        policy: SegmentationPolicy
    ) -> Bool {
        var precedingStart = matchStart
        while precedingStart > 0, !isWhitespace(codepoints[precedingStart - 1].value) {
            precedingStart -= 1
        }
        if policy.abbreviationTokens.contains(
            lowercase(codepoints, from: precedingStart, to: matchEnd)
        ) {
            return true
        }

        // Ordinal or list numbers such as "17." in "17. mai" never end a sentence.
        guard matchEnd - matchStart == 1, codepoints[matchStart].value == 0x2E else {
            return false
        }
        var digitStart = matchStart
        while digitStart > 0, isDigit(codepoints[digitStart - 1].value) {
            digitStart -= 1
        }
        let digitCount = matchStart - digitStart
        guard (1...3).contains(digitCount) else { return false }
        return digitStart == 0 || isWhitespace(codepoints[digitStart - 1].value)
    }

    static func splitParagraphSentences(
        _ codepoints: [TrackedCodepoint],
        policy: SegmentationPolicy
    ) -> [[TrackedCodepoint]] {
        var sentences: [[TrackedCodepoint]] = []
        var sentenceStart = 0
        var index = 0

        func emit(_ end: Int) {
            var start = sentenceStart
            while start < end, isWhitespace(codepoints[start].value) {
                start += 1
            }
            var last = end
            while last > start, isWhitespace(codepoints[last - 1].value) {
                last -= 1
            }
            if last > start {
                sentences.append(Array(codepoints[start..<last]))
            }
        }

        while index < codepoints.count {
            guard isTerminal(codepoints[index].value) else {
                index += 1
                continue
            }
            var matchEnd = index + 1
            while matchEnd < codepoints.count, isTerminal(codepoints[matchEnd].value) {
                matchEnd += 1
            }
            let matchStart = index
            index = matchEnd

            var remainderStart = matchEnd
            while remainderStart < codepoints.count,
                isWhitespace(codepoints[remainderStart].value)
            {
                remainderStart += 1
            }
            guard remainderStart < codepoints.count else { continue }
            let first = codepoints[remainderStart].value
            guard isUppercaseLetter(first) || isOpening(first) else { continue }
            // The terminal characters must be followed by whitespace.
            guard matchEnd < codepoints.count, isWhitespace(codepoints[matchEnd].value)
            else { continue }
            if isProtectedBoundary(
                codepoints, matchStart: matchStart, matchEnd: matchEnd, policy: policy
            ) {
                continue
            }
            emit(matchEnd)
            sentenceStart = matchEnd
        }
        if sentenceStart < codepoints.count {
            emit(codepoints.count)
        }
        return sentences
    }

    // MARK: - Tokenization

    struct RawToken {
        var start: Int // codepoint indices
        var end: Int
    }

    static func matches(
        _ codepoints: [TrackedCodepoint], at start: Int, ascii: String
    ) -> Bool {
        let expected = Array(ascii.unicodeScalars)
        guard start + expected.count <= codepoints.count else { return false }
        for offset in expected.indices
        where codepoints[start + offset].value != expected[offset].value {
            return false
        }
        return true
    }

    static func tryUrl(
        _ codepoints: [TrackedCodepoint], start: Int, end: inout Int
    ) -> Bool {
        guard
            matches(codepoints, at: start, ascii: "http://")
                || matches(codepoints, at: start, ascii: "https://")
                || matches(codepoints, at: start, ascii: "www.")
        else { return false }
        end = start
        while end < codepoints.count, !isWhitespace(codepoints[end].value) {
            end += 1
        }
        return true
    }

    static func tryEmail(
        _ codepoints: [TrackedCodepoint], start: Int, end: inout Int
    ) -> Bool {
        // [^\s@]+ @ [^\s@]+ . \w+
        var index = start
        while index < codepoints.count, !isWhitespace(codepoints[index].value),
            codepoints[index].value != 0x40
        {
            index += 1
        }
        guard index > start, index < codepoints.count, codepoints[index].value == 0x40
        else { return false }
        index += 1
        // The domain must contain a dot followed by word characters; scan the
        // remaining non-space, non-@ run and verify that shape.
        let domainStart = index
        while index < codepoints.count, !isWhitespace(codepoints[index].value),
            codepoints[index].value != 0x40
        {
            index += 1
        }
        var matchEnd = 0
        var position = domainStart
        while position < index {
            if codepoints[position].value == 0x2E {
                var wordEnd = position + 1
                while wordEnd < index, isWordCharacter(codepoints[wordEnd].value) {
                    wordEnd += 1
                }
                if wordEnd > position + 1 {
                    matchEnd = wordEnd
                }
            }
            position += 1
        }
        guard matchEnd > domainStart else { return false }
        end = matchEnd
        return true
    }

    static func consumeNumber(_ codepoints: [TrackedCodepoint], start: Int) -> Int {
        // \d+ ( [.,:/-] \d+ )*
        var index = start
        while index < codepoints.count, isDigit(codepoints[index].value) {
            index += 1
        }
        while index + 1 < codepoints.count {
            let separator = codepoints[index].value
            let isSeparator = separator == 0x2E || separator == 0x2C || separator == 0x3A
                || separator == 0x2F || separator == 0x2D
            guard isSeparator, isDigit(codepoints[index + 1].value) else { break }
            index += 1
            while index < codepoints.count, isDigit(codepoints[index].value) {
                index += 1
            }
        }
        return index
    }

    static func consumeWordRun(_ codepoints: [TrackedCodepoint], start: Int) -> Int {
        var index = start
        while index < codepoints.count, isWordCharacter(codepoints[index].value) {
            index += 1
        }
        return index
    }

    // \w+ (\.\w+)+ \.?   — dotted abbreviations such as "f.eks." stay one token.
    static func tryDottedAbbreviation(
        _ codepoints: [TrackedCodepoint], start: Int, end: inout Int
    ) -> Bool {
        var index = consumeWordRun(codepoints, start: start)
        guard index > start else { return false }
        var groups = 0
        while index < codepoints.count, codepoints[index].value == 0x2E {
            let wordEnd = consumeWordRun(codepoints, start: index + 1)
            if wordEnd == index + 1 { break }
            index = wordEnd
            groups += 1
        }
        guard groups > 0 else { return false }
        if index < codepoints.count, codepoints[index].value == 0x2E {
            index += 1
        }
        end = index
        return true
    }

    // \w+ ([-'’]\w+)*
    static func consumeWord(_ codepoints: [TrackedCodepoint], start: Int) -> Int {
        var index = consumeWordRun(codepoints, start: start)
        while index + 1 < codepoints.count {
            let connector = codepoints[index].value
            let isConnector = connector == 0x2D || connector == 0x27 || connector == 0x2019
            guard isConnector, isWordCharacter(codepoints[index + 1].value) else { break }
            index = consumeWordRun(codepoints, start: index + 1)
        }
        return index
    }

    static func scanRawTokens(_ codepoints: [TrackedCodepoint]) -> [RawToken] {
        var tokens: [RawToken] = []
        var position = 0

        while position < codepoints.count {
            if isWhitespace(codepoints[position].value) {
                position += 1
                continue
            }
            var end = 0
            if tryUrl(codepoints, start: position, end: &end) {
            } else if tryEmail(codepoints, start: position, end: &end) {
            } else if isDigit(codepoints[position].value) {
                end = consumeNumber(codepoints, start: position)
            } else if tryDottedAbbreviation(codepoints, start: position, end: &end) {
            } else if isWordCharacter(codepoints[position].value) {
                end = consumeWord(codepoints, start: position)
            } else {
                end = position + 1
            }
            tokens.append(RawToken(start: position, end: end))
            position = end
        }
        return tokens
    }

    // MARK: - Source ranges

    // The token's origin: adjacent codepoint origins merge into one
    // fragment, while origins separated by removed input (a popped hyphen,
    // a collapsed line break) stay separate fragments.
    static func coalesceSourceRanges(
        _ codepoints: [TrackedCodepoint], from start: Int, to end: Int
    ) -> [Utf8ByteRange] {
        var ranges: [Utf8ByteRange] = []
        for index in start..<end {
            let codepoint = codepoints[index]
            if codepoint.isSynthesized { continue }
            if let last = ranges.last, last.end == codepoint.sourceStart {
                ranges[ranges.count - 1] = Utf8ByteRange(
                    start: last.start, end: codepoint.sourceEnd
                )
            } else {
                ranges.append(
                    Utf8ByteRange(start: codepoint.sourceStart, end: codepoint.sourceEnd)
                )
            }
        }
        return ranges
    }

    static func gapIsWhitespace(_ bytes: [UInt8], from start: Int, to end: Int) -> Bool {
        decodeUtf8(Array(bytes[start..<end])).allSatisfy { isWhitespace($0.value) }
    }

    // Sentence ranges cover every token fragment: fragments whose gap in
    // the original text is pure whitespace share one range; gaps containing
    // removed non-whitespace content split the sentence into several ranges.
    static func sentenceSourceRanges(
        _ tokenSourceRanges: [[Utf8ByteRange]], originalBytes: [UInt8]
    ) -> [Utf8ByteRange] {
        var ranges: [Utf8ByteRange] = []
        for token in tokenSourceRanges {
            for fragment in token {
                if let last = ranges.last, last.end <= fragment.start,
                    gapIsWhitespace(originalBytes, from: last.end, to: fragment.start)
                {
                    ranges[ranges.count - 1] = Utf8ByteRange(
                        start: last.start, end: fragment.end
                    )
                } else {
                    ranges.append(fragment)
                }
            }
        }
        return ranges
    }

    static func tokenizeWithSpacing(
        _ codepoints: [TrackedCodepoint],
        policy: SegmentationPolicy,
        originalBytes: [UInt8]
    ) -> PretokenizedSentence {
        let raw = scanRawTokens(codepoints)

        var tokens: [String] = []
        var tokenSourceRanges: [[Utf8ByteRange]] = []
        var spans: [(start: Int, end: Int)] = []

        var index = 0
        while index < raw.count {
            let token = raw[index]
            let periodIsAttached = index + 1 < raw.count
                && raw[index + 1].end - raw[index + 1].start == 1
                && codepoints[raw[index + 1].start].value == 0x2E
                && raw[index + 1].start == token.end
            var keepsPeriod = false
            if periodIsAttached {
                let tokenText = text(of: codepoints, from: token.start, to: token.end)
                let tokenLength = token.end - token.start
                var allDigits = tokenLength <= 3
                var position = token.start
                while allDigits, position < token.end {
                    allDigits = isDigit(codepoints[position].value)
                    position += 1
                }
                keepsPeriod = policy.abbreviationTokens.contains(lowercase(tokenText) + ".")
                    || (allDigits && index + 2 < raw.count)
            }
            let tokenEnd = keepsPeriod ? raw[index + 1].end : token.end
            tokens.append(text(of: codepoints, from: token.start, to: tokenEnd))
            tokenSourceRanges.append(
                coalesceSourceRanges(codepoints, from: token.start, to: tokenEnd)
            )
            spans.append((token.start, tokenEnd))
            index += keepsPeriod ? 2 : 1
        }

        var hasSpaceBefore: [Bool] = []
        for position in tokens.indices {
            hasSpaceBefore.append(
                position == 0 ? false : spans[position - 1].end < spans[position].start
            )
        }
        return PretokenizedSentence(
            tokens: tokens,
            hasSpaceBefore: hasSpaceBefore,
            tokenSourceRanges: tokenSourceRanges,
            sourceRanges: sentenceSourceRanges(tokenSourceRanges, originalBytes: originalBytes)
        )
    }

    // MARK: - Chunking

    // Clip the parent's sentence ranges to the covering interval of one
    // chunk's token fragments; boundaries stay codepoint boundaries because
    // both inputs are codepoint boundaries of the same original text.
    static func clipSentenceRanges(
        _ ranges: [Utf8ByteRange], coverStart: Int, coverEnd: Int
    ) -> [Utf8ByteRange] {
        var clipped: [Utf8ByteRange] = []
        for range in ranges {
            let start = max(range.start, coverStart)
            let end = min(range.end, coverEnd)
            if start < end {
                clipped.append(Utf8ByteRange(start: start, end: end))
            }
        }
        return clipped
    }

    private static func appendChunks(
        to sentences: inout [PretokenizedSentence],
        sentence: PretokenizedSentence,
        maximumTokenCount: Int
    ) {
        if sentence.tokens.count <= maximumTokenCount {
            if !sentence.tokens.isEmpty { sentences.append(sentence) }
            return
        }
        let hasMapping = !sentence.tokenSourceRanges.isEmpty
        var start = 0
        while start < sentence.tokens.count {
            let end = min(start + maximumTokenCount, sentence.tokens.count)
            var spacing = Array(sentence.hasSpaceBefore[start..<end])
            spacing[0] = false
            var tokenRanges: [[Utf8ByteRange]] = []
            var sentenceRanges: [Utf8ByteRange] = []
            if hasMapping {
                tokenRanges = Array(sentence.tokenSourceRanges[start..<end])
                sentenceRanges = clipSentenceRanges(
                    sentence.sourceRanges,
                    coverStart: tokenRanges.first!.first!.start,
                    coverEnd: tokenRanges.last!.last!.end
                )
            }
            sentences.append(
                PretokenizedSentence(
                    tokens: Array(sentence.tokens[start..<end]),
                    hasSpaceBefore: spacing,
                    tokenSourceRanges: tokenRanges,
                    sourceRanges: sentenceRanges
                )
            )
            start = end
        }
    }
}
