import Foundation

/// One encoded sentence: subword IDs plus the word-to-subword alignment the
/// model's pooling consumes.
public struct EncodedSentence: Equatable, Sendable {
    public let inputIds: [Int]
    public let firstSubwordIndices: [Int]
    public let subwordEndIndices: [Int]
}

// MARK: - Normalization

/// A model's per-word normalization, applied before the byte-level mapping.
/// Selected from the tokenizer definition; see ``SubwordTokenizer``.
protocol SubwordNormalizer: Sendable {
    func normalize(_ word: String) -> String
}

/// NFKC (NorBERT4): fold the compatibility characters that occur in Western
/// prose. The Norwegian normalizer's newline Replace rules act on newlines,
/// which the runtime segmentation has already resolved, but a pre-tokenizer
/// word could still contain one, so they are honored here too.
struct CompatibilityNormalizer: SubwordNormalizer {
    func normalize(_ word: String) -> String {
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
}

/// NFC (ModernBERT): the artifact's words already arrive canonically composed,
/// so this is the identity — no compatibility characters to fold.
struct CanonicalNormalizer: SubwordNormalizer {
    func normalize(_ word: String) -> String { word }
}

// MARK: - Pre-tokenization

/// A model's pre-tokenizer: it splits a normalized word into the units BPE
/// operates on. Selected from the tokenizer definition; see ``SubwordTokenizer``.
protocol SubwordPreTokenizer: Sendable {
    func split(_ text: String) -> [String]
}

/// Applies a pre-tokenizer regex, emitting each match as one piece; any span
/// the regex leaves between matches is preserved as its own piece.
private func regexSplitPieces(_ text: String, pattern: NSRegularExpression) -> [String] {
    guard !text.isEmpty else { return [] }
    let range = NSRange(text.startIndex..., in: text)
    var pieces: [String] = []
    var cursor = text.startIndex
    for match in pattern.matches(in: text, range: range) {
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

/// ModernBERT's byte-level pre-tokenizer. Its GPT-2 regex is implicit in the
/// ByteLevel type (Hugging Face does not store it in the file), so the adapter
/// carries it: whole letter runs, whole digit runs, the lowercase contraction
/// suffixes, and a single attached leading space.
struct ByteLevelPreTokenizer: SubwordPreTokenizer {
    // swiftlint:disable:next force_try
    private static let gpt2Pattern = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    )

    func split(_ text: String) -> [String] {
        regexSplitPieces(text, pattern: Self.gpt2Pattern)
    }
}

/// NorBERT4's pre-tokenizer: a Split step whose regex — which breaks letter
/// runs at the case boundary and emits each digit on its own — travels in the
/// artifact, followed by the byte-level map.
struct NorbertPreTokenizer: SubwordPreTokenizer {
    let pattern: NSRegularExpression

    func split(_ text: String) -> [String] {
        regexSplitPieces(text, pattern: pattern)
    }
}

// MARK: - Tokenizer

/// Native byte-level BPE tokenizer executing the artifact's
/// `vocabulary.json` definition.
///
/// PrismKit ships its own subword tokenizer so weak target devices never pay
/// for a general-purpose tokenization framework; consumers may still opt
/// into a Hugging Face runtime, which reads the same file. Normalization,
/// pre-tokenization, and the special-token template are the points where a
/// model's convention enters: each is read from the definition and served by a
/// per-model adapter (NorBERT4's NFKC + case-splitting Split regex; ModernBERT's
/// NFC + byte-level GPT-2 regex). Parity with the reference implementation is
/// enforced token by token in the test suite.
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
    private let normalizer: any SubwordNormalizer
    private let preTokenizer: any SubwordPreTokenizer
    // Special tokens wrapping each sequence, read from the post-processor
    // template: NorBERT4 has an "<s>" prefix and no suffix; ModernBERT has a
    // "[CLS]" prefix and "[SEP]" suffix.
    private let prefixIds: [Int]
    private let suffixIds: [Int]

    public init(vocabularyURL: URL) throws {
        let data = try Data(contentsOf: vocabularyURL)
        let definition = try JSONDecoder().decode(TokenizerDefinition.self, from: data)
        let model = definition.model

        vocabulary = model.vocab
        ignoreMerges = model.ignoreMerges ?? false

        // Byte-level BPE never emits the unknown token; read it only when the
        // model declares one (ModernBERT's is null), otherwise default to 0.
        if let token = model.unkToken, let identifier = model.vocab[token] {
            unknownId = identifier
        } else {
            unknownId = 0
        }

        var ranks: [MergePair: Int] = [:]
        ranks.reserveCapacity(model.merges.count)
        for (rank, merge) in model.merges.enumerated() {
            ranks[MergePair(left: merge.left, right: merge.right)] = rank
        }
        mergeRanks = ranks

        byteToUnicode = Self.buildByteToUnicode()
        normalizer = Self.makeNormalizer(definition.normalizer)
        preTokenizer = try Self.makePreTokenizer(definition.preTokenizer)
        (prefixIds, suffixIds) = try Self.templateTokens(definition.postProcessor)
    }

    /// Encode UD word tokens into subword IDs with word alignment.
    public func encode(_ sentence: PretokenizedSentence) -> EncodedSentence {
        var inputIds: [Int] = prefixIds
        var firstSubwordIndices: [Int] = []
        var subwordEndIndices: [Int] = []

        for (index, token) in sentence.tokens.enumerated() {
            let word = sentence.hasSpaceBefore[index] ? " " + token : token
            let subwordIds = encodeWord(word)
            firstSubwordIndices.append(inputIds.count)
            inputIds.append(contentsOf: subwordIds)
            subwordEndIndices.append(inputIds.count)
        }
        inputIds.append(contentsOf: suffixIds)
        return EncodedSentence(
            inputIds: inputIds,
            firstSubwordIndices: firstSubwordIndices,
            subwordEndIndices: subwordEndIndices
        )
    }

    func encodeWord(_ word: String) -> [Int] {
        var identifiers: [Int] = []
        for piece in preTokenizer.split(normalizer.normalize(word)) {
            let mapped = byteLevelEncode(piece)
            identifiers.append(contentsOf: bytePairEncode(mapped))
        }
        return identifiers
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

    // MARK: Adapter selection

    /// NFKC folds compatibility characters; everything else (NFC, or none) is
    /// the identity on already-composed input.
    private static func makeNormalizer(_ node: NormalizerNode?) -> any SubwordNormalizer {
        var types: [String] = []
        node?.collectTypes(into: &types)
        return types.contains("NFKC") ? CompatibilityNormalizer() : CanonicalNormalizer()
    }

    /// A bare byte-level map is ModernBERT's GPT-2 pre-tokenizer; a Sequence
    /// carrying a Split step is NorBERT4's, whose regex travels in the file. An
    /// unrecognized configuration is a hard error rather than a silent
    /// mis-tokenization of a new model.
    private static func makePreTokenizer(
        _ node: PreTokenizerNode
    ) throws -> any SubwordPreTokenizer {
        if node.type == "ByteLevel" {
            return ByteLevelPreTokenizer()
        }
        if node.type == "Sequence",
            let split = node.pretokenizers?.first(where: { $0.type == "Split" }),
            let regex = split.pattern?.regex
        {
            return NorbertPreTokenizer(pattern: try NSRegularExpression(pattern: regex))
        }
        throw PrismError.invalidArtifact(
            "Unsupported tokenizer pre_tokenizer: \(node.type)."
        )
    }

    /// The special tokens the TemplateProcessing post-processor wraps each
    /// sequence with, split into the ids before and after the sequence slot.
    private static func templateTokens(
        _ node: PostProcessorNode
    ) throws -> (prefix: [Int], suffix: [Int]) {
        guard node.type == "TemplateProcessing" else {
            throw PrismError.invalidArtifact(
                "Unsupported tokenizer post_processor: \(node.type)."
            )
        }
        var prefix: [Int] = []
        var suffix: [Int] = []
        var afterSequence = false
        for piece in node.single {
            if piece.sequence != nil {
                afterSequence = true
            } else if let token = piece.specialToken?.id,
                let ids = node.specialTokens[token]?.ids
            {
                if afterSequence {
                    suffix.append(contentsOf: ids)
                } else {
                    prefix.append(contentsOf: ids)
                }
            }
        }
        return (prefix, suffix)
    }
}

// MARK: - Tokenizer definition decoding

/// The subset of the Hugging Face `tokenizer.json` the native tokenizer reads.
private struct TokenizerDefinition: Decodable {
    let model: Model
    let normalizer: NormalizerNode?
    let preTokenizer: PreTokenizerNode
    let postProcessor: PostProcessorNode

    enum CodingKeys: String, CodingKey {
        case model, normalizer
        case preTokenizer = "pre_tokenizer"
        case postProcessor = "post_processor"
    }

    struct Model: Decodable {
        let vocab: [String: Int]
        let merges: [MergeEntry]
        let unkToken: String?
        let ignoreMerges: Bool?

        enum CodingKeys: String, CodingKey {
            case vocab, merges
            case unkToken = "unk_token"
            case ignoreMerges = "ignore_merges"
        }
    }

    struct MergeEntry: Decodable {
        let left: String
        let right: String
        init(from decoder: Decoder) throws {
            var container = try decoder.unkeyedContainer()
            left = try container.decode(String.self)
            right = try container.decode(String.self)
        }
    }
}

/// A normalizer node: a bare type ("NFC", "NFKC") or a Sequence of them.
private struct NormalizerNode: Decodable {
    let type: String
    let normalizers: [NormalizerNode]?

    func collectTypes(into types: inout [String]) {
        if type == "Sequence" {
            for inner in normalizers ?? [] {
                inner.collectTypes(into: &types)
            }
        } else {
            types.append(type)
        }
    }
}

/// A pre-tokenizer node: a bare type ("ByteLevel") or a Sequence carrying a
/// Split step whose regex pattern travels in `pattern.Regex`.
private struct PreTokenizerNode: Decodable {
    let type: String
    let pretokenizers: [PreTokenizerNode]?
    let pattern: Pattern?

    struct Pattern: Decodable {
        let regex: String?
        enum CodingKeys: String, CodingKey { case regex = "Regex" }
    }
}

/// A TemplateProcessing post-processor: `single` is the per-sequence template,
/// its SpecialToken ids resolved through `special_tokens`.
private struct PostProcessorNode: Decodable {
    let type: String
    let single: [TemplatePiece]
    let specialTokens: [String: SpecialToken]

    enum CodingKeys: String, CodingKey {
        case type, single
        case specialTokens = "special_tokens"
    }

    struct SpecialToken: Decodable {
        let ids: [Int]
    }

    struct TemplatePiece: Decodable {
        let specialToken: TokenRef?
        let sequence: TokenRef?

        enum CodingKeys: String, CodingKey {
            case specialToken = "SpecialToken"
            case sequence = "Sequence"
        }

        struct TokenRef: Decodable {
            let id: String
        }
    }
}
