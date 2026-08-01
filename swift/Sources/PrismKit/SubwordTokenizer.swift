import Foundation

/// One encoded sentence: subword IDs plus the word-to-subword alignment the
/// model's pooling consumes.
public struct EncodedSentence: Equatable, Sendable {
    public let inputIds: [Int]
    public let firstSubwordIndices: [Int]
    public let subwordEndIndices: [Int]
}

/// Native byte-level BPE tokenizer executing the artifact's
/// `vocabulary.json` definition.
///
/// PrismKit ships its own subword tokenizer so weak target devices never pay
/// for a general-purpose tokenization framework; consumers may still opt
/// into a Hugging Face runtime, which reads the same file. Parity with the
/// reference implementation is enforced token by token in the test suite.
public struct SubwordTokenizer: Sendable {
    private struct MergePair: Hashable {
        let left: String
        let right: String
    }

    private let vocabulary: [String: Int]
    private let mergeRanks: [MergePair: Int]
    private let ignoreMerges: Bool
    private let unknownId: Int
    private let byteToUnicode: [Character]
    private let splitPattern: NSRegularExpression

    public let beginOfSequenceId: Int

    public init(vocabularyURL: URL) throws {
        struct MergeEntry: Decodable {
            let left: String
            let right: String
            init(from decoder: Decoder) throws {
                var container = try decoder.unkeyedContainer()
                left = try container.decode(String.self)
                right = try container.decode(String.self)
            }
        }
        struct Model: Decodable {
            let vocab: [String: Int]
            let merges: [MergeEntry]
            let unkToken: String
            let ignoreMerges: Bool?
            enum CodingKeys: String, CodingKey {
                case vocab, merges
                case unkToken = "unk_token"
                case ignoreMerges = "ignore_merges"
            }
        }
        struct Definition: Decodable {
            let model: Model
        }

        let data = try Data(contentsOf: vocabularyURL)
        let definition = try JSONDecoder().decode(Definition.self, from: data)

        vocabulary = definition.model.vocab
        ignoreMerges = definition.model.ignoreMerges ?? false
        guard let unknown = definition.model.vocab[definition.model.unkToken] else {
            throw PrismError.invalidArtifact("Tokenizer misses its unknown token.")
        }
        unknownId = unknown
        guard let begin = definition.model.vocab["<s>"] else {
            throw PrismError.invalidArtifact("Tokenizer misses the <s> token.")
        }
        beginOfSequenceId = begin

        var ranks: [MergePair: Int] = [:]
        ranks.reserveCapacity(definition.model.merges.count)
        for (rank, merge) in definition.model.merges.enumerated() {
            ranks[MergePair(left: merge.left, right: merge.right)] = rank
        }
        mergeRanks = ranks

        byteToUnicode = Self.buildByteToUnicode()
        splitPattern = try NSRegularExpression(
            pattern: #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"#
                + #"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"#
                + #"|\p{N}"#
                + #"| ?[^\s\p{L}\p{N}]+[\r\n/]*"#
                + #"|\s*[\r\n]+"#
                + #"|\s+(?!\S)"#
                + #"|\s+"#
        )
    }

    /// Encode UD word tokens into subword IDs with word alignment.
    public func encode(_ sentence: PretokenizedSentence) -> EncodedSentence {
        var inputIds: [Int] = [beginOfSequenceId]
        var firstSubwordIndices: [Int] = []
        var subwordEndIndices: [Int] = []

        for (index, token) in sentence.tokens.enumerated() {
            let word = sentence.hasSpaceBefore[index] ? " " + token : token
            let subwordIds = encodeWord(word)
            firstSubwordIndices.append(inputIds.count)
            inputIds.append(contentsOf: subwordIds)
            subwordEndIndices.append(inputIds.count)
        }
        return EncodedSentence(
            inputIds: inputIds,
            firstSubwordIndices: firstSubwordIndices,
            subwordEndIndices: subwordEndIndices
        )
    }

    func encodeWord(_ word: String) -> [Int] {
        var identifiers: [Int] = []
        for piece in splitPieces(normalize(word)) {
            let mapped = byteLevelEncode(piece)
            identifiers.append(contentsOf: bytePairEncode(mapped))
        }
        return identifiers
    }

    private func normalize(_ word: String) -> String {
        var text = word.precomposedStringWithCompatibilityMapping
        if text.contains("\n") {
            text = text.replacingOccurrences(of: "\n", with: "\n ")
            text = text.replacingOccurrences(
                of: " *\n",
                with: "\n",
                options: .regularExpression
            )
        }
        return text
    }

    private func splitPieces(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        var pieces: [String] = []
        var cursor = text.startIndex
        for match in splitPattern.matches(in: text, range: range) {
            guard let matchRange = Range(match.range, in: text) else { continue }
            if cursor < matchRange.lowerBound {
                pieces.append(String(text[cursor..<matchRange.lowerBound]))
            }
            pieces.append(String(text[matchRange]))
            cursor = matchRange.upperBound
        }
        if cursor < text.endIndex {
            pieces.append(String(text[cursor...]))
        }
        return pieces
    }

    private func byteLevelEncode(_ piece: String) -> String {
        var mapped = ""
        mapped.reserveCapacity(piece.utf8.count)
        for byte in piece.utf8 {
            mapped.append(byteToUnicode[Int(byte)])
        }
        return mapped
    }

    private func bytePairEncode(_ mapped: String) -> [Int] {
        if ignoreMerges, let identifier = vocabulary[mapped] {
            return [identifier]
        }

        var symbols = mapped.map { String($0) }
        guard symbols.count > 1 else {
            return symbols.map { vocabulary[$0] ?? unknownId }
        }

        while true {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(symbols.count - 1) {
                let pair = MergePair(left: symbols[index], right: symbols[index + 1])
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }
            symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
            if symbols.count == 1 { break }
        }
        return symbols.map { vocabulary[$0] ?? unknownId }
    }

    private static func buildByteToUnicode() -> [Character] {
        var byteValues: [Int] = []
        byteValues.append(contentsOf: 33...126)
        byteValues.append(contentsOf: 161...172)
        byteValues.append(contentsOf: 174...255)
        var scalars = byteValues
        var next = 0
        for byte in 0...255 where !byteValues.contains(byte) {
            byteValues.append(byte)
            scalars.append(256 + next)
            next += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (byte, scalar) in zip(byteValues, scalars) {
            table[byte] = Character(UnicodeScalar(scalar)!)
        }
        return table
    }
}
