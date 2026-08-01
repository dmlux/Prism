import ExecuTorch
import Foundation

/// One tagged token with calibrated confidences per decision.
public struct TaggedToken: Sendable {
    public let text: String
    public let hasSpaceBefore: Bool
    public let upos: String
    public let uposConfidence: Double
    public let features: [String: [String]]
    public let featureConfidences: [String: Double]
    public let lemma: String
    public let lemmaConfidence: Double
}

public struct TaggedSentence: Sendable {
    public let tokens: [TaggedToken]
}

/// Frozen-artifact tagger: raw text or word tokens in, decisions plus
/// calibrated confidences out. The program already contains the complete
/// decoding policy, so this class only assembles fixed-shape batches and
/// applies argmax, the 0.5 threshold, and the lemma edit rules.
public final class PrismTagger {
    private let artifact: PrismArtifact
    private let program: ArtifactProgram
    private let module: Module
    private let tokenizer: SubwordTokenizer
    private let segmentationPolicy: SegmentationPolicy
    private let characterIds: [String: Int]
    private let maximumCharacterCount: Int
    private let paddingTokenId: Int

    private static let characterPaddingId = 0
    private static let characterUnknownId = 1
    private static let characterStartId = 2
    private static let characterEndId = 3
    private static let characterTruncationId = 4
    private static let firstLiteralCharacterId = 5

    public init(artifactURL: URL, device: ComputeDevice = .automatic) throws {
        artifact = try PrismArtifact(contentsOf: artifactURL)
        program = try artifact.program(for: device)
        module = Module(
            filePath: artifactURL.appendingPathComponent(program.fileName).path
        )
        try module.load()
        tokenizer = try SubwordTokenizer(
            vocabularyURL: artifactURL.appendingPathComponent(
                artifact.manifest.vocabularyFile
            )
        )
        segmentationPolicy = .norwegian(
            maximumTokenCount: program.shapes.tokenCount
        )
        var lookup: [String: Int] = [:]
        for (index, character) in (artifact.labels.characterVocabulary?.characters ?? [])
            .enumerated()
        {
            lookup[character] = Self.firstLiteralCharacterId + index
        }
        characterIds = lookup
        maximumCharacterCount = artifact.labels.maximumCharacterCount ?? 32
        paddingTokenId = artifact.manifest.tokenizer.paddingTokenId
    }

    /// Segment raw text with the runtime policy, then tag every sentence.
    public func tag(text: String) throws -> [TaggedSentence] {
        try tag(sentences: RuntimeSegmentation.segment(text, policy: segmentationPolicy))
    }

    /// Tag application-supplied word tokens (space assumed between words).
    public func tag(pretokenized: [[String]]) throws -> [TaggedSentence] {
        try tag(
            sentences: pretokenized.compactMap { tokens in
                tokens.isEmpty
                    ? nil
                    : PretokenizedSentence(
                        tokens: tokens,
                        hasSpaceBefore: [false] + Array(
                            repeating: true,
                            count: tokens.count - 1
                        )
                    )
            }
        )
    }

    public func tag(sentences: [PretokenizedSentence]) throws -> [TaggedSentence] {
        let prepared = sentences.flatMap {
            RuntimeSegmentation.chunk($0, maximumTokenCount: program.shapes.tokenCount)
        }
        var tagged: [TaggedSentence] = []
        var start = 0
        while start < prepared.count {
            let end = min(start + program.shapes.batchSize, prepared.count)
            tagged.append(contentsOf: try tagBatch(Array(prepared[start..<end])))
            start = end
        }
        return tagged
    }

    private func tagBatch(_ realSentences: [PretokenizedSentence]) throws -> [TaggedSentence] {
        let shapes = program.shapes
        var sentences = realSentences
        while sentences.count < shapes.batchSize {
            sentences.append(sentences[sentences.count - 1])
        }

        let encoded = sentences.map(tokenizer.encode)
        for sentence in encoded
        where sentence.inputIds.count > shapes.subwordCount {
            throw PrismError.invalidArtifact(
                "Sentence exceeds the program's subword capacity."
            )
        }

        var inputIds = [Int64](
            repeating: Int64(paddingTokenId),
            count: shapes.batchSize * shapes.subwordCount
        )
        var attentionMask = [Bool](
            repeating: false,
            count: shapes.batchSize * shapes.subwordCount
        )
        var firstIndices = [Int64](
            repeating: 0,
            count: shapes.batchSize * shapes.tokenCount
        )
        var endIndices = [Int64](
            repeating: 0,
            count: shapes.batchSize * shapes.tokenCount
        )
        var tokenMask = [Bool](
            repeating: false,
            count: shapes.batchSize * shapes.tokenCount
        )
        let characterCount = shapes.characterCount ?? maximumCharacterCount
        var characterIdsTensor = [Int64](
            repeating: Int64(Self.characterPaddingId),
            count: shapes.batchSize * shapes.tokenCount * characterCount
        )
        var characterMask = [Bool](
            repeating: false,
            count: shapes.batchSize * shapes.tokenCount * characterCount
        )

        for (row, sentence) in encoded.enumerated() {
            let subwordBase = row * shapes.subwordCount
            for (offset, identifier) in sentence.inputIds.enumerated() {
                inputIds[subwordBase + offset] = Int64(identifier)
                attentionMask[subwordBase + offset] = true
            }
            let tokenBase = row * shapes.tokenCount
            for token in 0..<sentence.firstSubwordIndices.count {
                firstIndices[tokenBase + token] = Int64(sentence.firstSubwordIndices[token])
                endIndices[tokenBase + token] = Int64(sentence.subwordEndIndices[token])
                tokenMask[tokenBase + token] = true
                let characterBase = (tokenBase + token) * characterCount
                let ids = encodeCharacters(sentences[row].tokens[token])
                for (offset, identifier) in ids.enumerated() {
                    characterIdsTensor[characterBase + offset] = Int64(identifier)
                    characterMask[characterBase + offset] = true
                }
            }
        }

        var values: [Value] = [
            Value(Tensor<Int64>(inputIds, shape: [shapes.batchSize, shapes.subwordCount])),
            Value(Tensor<Bool>(attentionMask, shape: [shapes.batchSize, shapes.subwordCount])),
            Value(Tensor<Int64>(firstIndices, shape: [shapes.batchSize, shapes.tokenCount])),
            Value(Tensor<Int64>(endIndices, shape: [shapes.batchSize, shapes.tokenCount])),
            Value(Tensor<Bool>(tokenMask, shape: [shapes.batchSize, shapes.tokenCount])),
        ]
        if shapes.characterCount != nil {
            values.append(
                Value(
                    Tensor<Int64>(
                        characterIdsTensor,
                        shape: [shapes.batchSize, shapes.tokenCount, characterCount]
                    )
                )
            )
            values.append(
                Value(
                    Tensor<Bool>(
                        characterMask,
                        shape: [shapes.batchSize, shapes.tokenCount, characterCount]
                    )
                )
            )
        }

        let outputs = try module.forward(values)
        var probabilities: [[Float]] = []
        for output in outputs {
            let tensor: Tensor<Float> = try output.tensor()
                ?? { throw PrismError.invalidArtifact("Program output is not a tensor.") }()
            probabilities.append(try tensor.scalars())
        }
        return try decode(realSentences, probabilities: probabilities)
    }

    private func decode(
        _ sentences: [PretokenizedSentence],
        probabilities: [[Float]]
    ) throws -> [TaggedSentence] {
        let shapes = program.shapes
        let schema = artifact.labels.schema
        let uposCount = schema.upos.labels.count
        let ruleCount = schema.lemmaRules.rules.count

        var tagged: [TaggedSentence] = []
        for (row, sentence) in sentences.enumerated() {
            var tokens: [TaggedToken] = []
            for tokenIndex in 0..<sentence.tokens.count {
                let flat = row * shapes.tokenCount + tokenIndex

                let uposRow = Array(
                    probabilities[0][(flat * uposCount)..<((flat + 1) * uposCount)]
                )
                let (uposIndex, uposConfidence) = argmax(uposRow)

                var features: [String: [String]] = [:]
                var featureConfidences: [String: Double] = [:]
                for (featureIndex, feature) in schema.morphology.features.enumerated() {
                    let output = probabilities[1 + featureIndex]
                    if feature.allowsMultipleValues {
                        let count = feature.values.count
                        let valueRow = Array(output[(flat * count)..<((flat + 1) * count)])
                        let selected = zip(feature.values, valueRow)
                            .filter { $0.1 > 0.5 }
                        if !selected.isEmpty {
                            features[feature.name] = selected.map(\.0)
                            featureConfidences[feature.name] = Double(
                                selected.map(\.1).min()!
                            )
                        }
                    } else {
                        let count = feature.values.count + 1
                        let valueRow = Array(output[(flat * count)..<((flat + 1) * count)])
                        let (valueIndex, confidence) = argmax(valueRow)
                        if valueIndex > 0 {
                            features[feature.name] = [feature.values[valueIndex - 1]]
                            featureConfidences[feature.name] = confidence
                        }
                    }
                }

                let lemmaRow = Array(
                    probabilities[probabilities.count - 1][
                        (flat * ruleCount)..<((flat + 1) * ruleCount)
                    ]
                )
                let (ruleIndex, lemmaConfidence) = argmax(lemmaRow)
                let tokenText = sentence.tokens[tokenIndex]
                let lemma = (try? schema.lemmaRules.rules[ruleIndex].apply(to: tokenText))
                    ?? tokenText

                tokens.append(
                    TaggedToken(
                        text: tokenText,
                        hasSpaceBefore: sentence.hasSpaceBefore[tokenIndex],
                        upos: schema.upos.labels[uposIndex],
                        uposConfidence: uposConfidence,
                        features: features,
                        featureConfidences: featureConfidences,
                        lemma: lemma,
                        lemmaConfidence: lemmaConfidence
                    )
                )
            }
            tagged.append(TaggedSentence(tokens: tokens))
        }
        return tagged
    }

    private func argmax(_ row: [Float]) -> (Int, Double) {
        var bestIndex = 0
        var bestValue = -Float.infinity
        for (index, value) in row.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = index
        }
        return (bestIndex, Double(bestValue))
    }

    private func encodeCharacters(_ token: String) -> [Int] {
        let normalized = token.precomposedStringWithCanonicalMapping
        let literalIds = normalized.unicodeScalars.map {
            characterIds[String($0)] ?? Self.characterUnknownId
        }
        let complete = [Self.characterStartId] + literalIds + [Self.characterEndId]
        if complete.count <= maximumCharacterCount {
            return complete
        }
        let retained = maximumCharacterCount - 3
        let prefixCount = (retained + 1) / 2
        let suffixCount = retained - prefixCount
        return [Self.characterStartId]
            + literalIds.prefix(prefixCount)
            + [Self.characterTruncationId]
            + literalIds.suffix(suffixCount)
            + [Self.characterEndId]
    }
}
