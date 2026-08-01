import Foundation

/// One sentence as UD-convention word tokens plus spacing information.
public struct PretokenizedSentence: Equatable, Sendable {
    public let tokens: [String]
    public let hasSpaceBefore: [Bool]

    public init(tokens: [String], hasSpaceBefore: [Bool]) {
        precondition(tokens.count == hasSpaceBefore.count)
        self.tokens = tokens
        self.hasSpaceBefore = hasSpaceBefore
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
public enum RuntimeSegmentation {
    private static let terminalCharacters: Set<Character> = [".", "!", "?", "…"]
    private static let openingCharacters: Set<Character> = ["«", "\"", "'", "(", "[", "„", "\u{201C}"]

    private static let missingSpacePattern = try! NSRegularExpression(
        pattern: "(?<=[a-zæøå])([.!?…])(?=[A-ZÆØÅ«\"'\\(\\[„\u{201C}])"
    )

    private static let tokenPattern = try! NSRegularExpression(
        pattern: #"(?:https?://|www\.)\S+"#
            + #"|[^\s@]+@[^\s@]+\.\w+"#
            + #"|\d+(?:[.,:/\-]\d+)*"#
            + #"|\w+(?:\.\w+)+\.?"#
            + #"|\w+(?:[-'’]\w+)*"#
            + #"|\S"#
    )

    /// Segment raw text without discarding content.
    public static func segment(
        _ text: String,
        policy: SegmentationPolicy
    ) -> [PretokenizedSentence] {
        var sentences: [PretokenizedSentence] = []
        let repaired = restoreMissingSentenceSpaces(text)
        for paragraph in mergeWrappedLines(repaired) {
            for sentenceText in splitParagraphSentences(paragraph, policy: policy) {
                guard let sentence = tokenizeWithSpacing(sentenceText, policy: policy)
                else { continue }
                sentences.append(
                    contentsOf: chunk(sentence, maximumTokenCount: policy.maximumTokenCount)
                )
            }
        }
        return sentences
    }

    /// Restore spaces lost after sentence punctuation ("veien.Et sekund").
    ///
    /// A lowercase letter, terminal punctuation, and an immediately following
    /// uppercase or opening character never form one token in Norwegian
    /// prose; abbreviation-protected boundaries are still consulted after the
    /// repair, so "f.eks.Dette" separates without a false sentence break.
    static func restoreMissingSentenceSpaces(_ text: String) -> String {
        missingSpacePattern.stringByReplacingMatches(
            in: text,
            range: NSRange(text.startIndex..., in: text),
            withTemplate: "$1 "
        )
    }

    /// Join wrapped lines into paragraphs without joining layout lines.
    static func mergeWrappedLines(_ text: String) -> [String] {
        var paragraphs: [String] = []
        var paragraph = ""

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.split(whereSeparator: { $0.isWhitespace })
                .joined(separator: " ")
            if line.isEmpty {
                if !paragraph.isEmpty { paragraphs.append(paragraph) }
                paragraph = ""
                continue
            }
            let continuesPrevious = !paragraph.isEmpty
                && !terminalCharacters.contains(paragraph.last!)
                && paragraph.last! != ":"
                && line.first!.isLowercase
            if continuesPrevious && paragraph.hasSuffix("-") {
                paragraph = String(paragraph.dropLast()) + line
            } else if continuesPrevious {
                paragraph += " " + line
            } else {
                if !paragraph.isEmpty { paragraphs.append(paragraph) }
                paragraph = line
            }
        }
        if !paragraph.isEmpty { paragraphs.append(paragraph) }
        return paragraphs
    }

    static func splitParagraphSentences(
        _ paragraph: String,
        policy: SegmentationPolicy
    ) -> [String] {
        let characters = Array(paragraph)
        var sentences: [String] = []
        var sentenceStart = 0
        var index = 0

        while index < characters.count {
            guard terminalCharacters.contains(characters[index]) else {
                index += 1
                continue
            }
            var matchEnd = index + 1
            while matchEnd < characters.count,
                terminalCharacters.contains(characters[matchEnd])
            {
                matchEnd += 1
            }
            let matchStart = index
            index = matchEnd

            var remainderStart = matchEnd
            while remainderStart < characters.count,
                characters[remainderStart].isWhitespace
            {
                remainderStart += 1
            }
            guard remainderStart < characters.count else { continue }
            let first = characters[remainderStart]
            guard first.isUppercase || openingCharacters.contains(first) else { continue }
            // The terminal characters must be followed by whitespace.
            guard matchEnd < characters.count, characters[matchEnd].isWhitespace
            else { continue }
            if isProtectedBoundary(
                characters,
                matchStart: matchStart,
                matchEnd: matchEnd,
                policy: policy
            ) {
                continue
            }
            let sentence = String(characters[sentenceStart..<matchEnd])
                .trimmingCharacters(in: .whitespaces)
            if !sentence.isEmpty { sentences.append(sentence) }
            sentenceStart = matchEnd
        }

        if sentenceStart < characters.count {
            let tail = String(characters[sentenceStart...])
                .trimmingCharacters(in: .whitespaces)
            if !tail.isEmpty { sentences.append(tail) }
        }
        return sentences
    }

    private static func isProtectedBoundary(
        _ characters: [Character],
        matchStart: Int,
        matchEnd: Int,
        policy: SegmentationPolicy
    ) -> Bool {
        var precedingStart = matchStart
        while precedingStart > 0, !characters[precedingStart - 1].isWhitespace {
            precedingStart -= 1
        }
        let precedingWord = String(characters[precedingStart..<matchEnd]).lowercased()
        if policy.abbreviationTokens.contains(precedingWord) { return true }

        // Ordinal or list numbers such as "17." in "17. mai" never end a sentence.
        guard matchEnd - matchStart == 1, characters[matchStart] == "." else {
            return false
        }
        var digitStart = matchStart
        while digitStart > 0, isDecimalDigit(characters[digitStart - 1]) {
            digitStart -= 1
        }
        let digitCount = matchStart - digitStart
        guard (1...3).contains(digitCount) else { return false }
        return digitStart == 0 || characters[digitStart - 1].isWhitespace
    }

    private static func isDecimalDigit(_ character: Character) -> Bool {
        character.unicodeScalars.count == 1
            && CharacterSet.decimalDigits.contains(character.unicodeScalars.first!)
    }

    static func tokenizeWithSpacing(
        _ sentenceText: String,
        policy: SegmentationPolicy
    ) -> PretokenizedSentence? {
        let range = NSRange(sentenceText.startIndex..., in: sentenceText)
        let matches = tokenPattern.matches(in: sentenceText, range: range)
        guard !matches.isEmpty else { return nil }

        var rawTokens: [String] = []
        var rawSpans: [(start: Int, end: Int)] = []
        for match in matches {
            guard let matchRange = Range(match.range, in: sentenceText) else { continue }
            rawTokens.append(String(sentenceText[matchRange]))
            rawSpans.append((match.range.location, match.range.location + match.range.length))
        }

        // Reattach the period of listed abbreviations, and of short ordinal
        // numbers such as "17." in "17. mai", to match the UD token
        // convention. A sentence-final period after a number remains its own
        // token.
        var tokens: [String] = []
        var spans: [(start: Int, end: Int)] = []
        var index = 0
        while index < rawTokens.count {
            let token = rawTokens[index]
            let span = rawSpans[index]
            let periodIsAttached = index + 1 < rawTokens.count
                && rawTokens[index + 1] == "."
                && rawSpans[index + 1].start == span.end
            let isShortNumber = token.count <= 3
                && !token.isEmpty
                && token.allSatisfy { isDecimalDigit($0) }
            let keepsPeriod = periodIsAttached
                && (policy.abbreviationTokens.contains(token.lowercased() + ".")
                    || (isShortNumber && index + 2 < rawTokens.count))
            if keepsPeriod {
                tokens.append(token + ".")
                spans.append((span.start, rawSpans[index + 1].end))
                index += 2
                continue
            }
            tokens.append(token)
            spans.append(span)
            index += 1
        }

        var hasSpaceBefore: [Bool] = []
        for position in tokens.indices {
            hasSpaceBefore.append(
                position == 0 ? false : spans[position - 1].end < spans[position].start
            )
        }
        return PretokenizedSentence(tokens: tokens, hasSpaceBefore: hasSpaceBefore)
    }

    static func chunk(
        _ sentence: PretokenizedSentence,
        maximumTokenCount: Int
    ) -> [PretokenizedSentence] {
        guard sentence.tokens.count > maximumTokenCount else { return [sentence] }
        var chunks: [PretokenizedSentence] = []
        var start = 0
        while start < sentence.tokens.count {
            let end = min(start + maximumTokenCount, sentence.tokens.count)
            var spacing = Array(sentence.hasSpaceBefore[start..<end])
            spacing[0] = false
            chunks.append(
                PretokenizedSentence(
                    tokens: Array(sentence.tokens[start..<end]),
                    hasSpaceBefore: spacing
                )
            )
            start = end
        }
        return chunks
    }
}
