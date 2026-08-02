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
///
/// Artifacts may ship several fixed-shape programs; sentences are sorted by
/// length and every batch runs on the smallest program it fits into, so
/// short sentences never pay the padding cost of the largest shapes.
public final class PrismTagger {
    private let artifact: PrismArtifact
    private let programs: [ArtifactProgram]
    private var modules: [String: Module] = [:]
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
        // Six threads is the measured sweet spot for the small fixed-shape
        // batches; an explicit ComputeThreads.setThreadCount wins.
        ComputeThreads.installDefault(6)
        artifact = try PrismArtifact(contentsOf: artifactURL)
        programs = try artifact.programs(for: device)
        tokenizer = try SubwordTokenizer(
            vocabularyURL: artifactURL.appendingPathComponent(
                artifact.manifest.vocabularyFile
            )
        )
        segmentationPolicy = .norwegian(
            maximumTokenCount: programs.last!.shapes.tokenCount
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
        let largest = programs.last!.shapes
        let prepared = sentences.flatMap {
            RuntimeSegmentation.chunk($0, maximumTokenCount: largest.tokenCount)
        }
        let encoded = prepared.map(tokenizer.encode)
        for sentence in encoded where sentence.inputIds.count > largest.subwordCount {
            throw PrismError.invalidArtifact(
                "Sentence exceeds the largest program's subword capacity."
            )
        }

        // Length-sorted batching keeps short sentences together so their
        // batches qualify for the smallest lowered program.
        let order = prepared.indices.sorted {
            encoded[$0].inputIds.count < encoded[$1].inputIds.count
        }
        var tagged = [TaggedSentence?](repeating: nil, count: prepared.count)
        let batchSize = largest.batchSize
        var start = 0
        while start < order.count {
            let end = min(start + batchSize, order.count)
            let batchIndices = Array(order[start..<end])
            let batchSentences = batchIndices.map { prepared[$0] }
            let batchEncoded = batchIndices.map { encoded[$0] }
            let program = smallestFittingProgram(for: batchEncoded, in: batchSentences)
            let decoded = try tagBatch(
                batchSentences,
                encoded: batchEncoded,
                program: program
            )
            for (position, index) in batchIndices.enumerated() {
                tagged[index] = decoded[position]
            }
            start = end
        }
        return tagged.map { $0! }
    }

    private func smallestFittingProgram(
        for encoded: [EncodedSentence],
        in sentences: [PretokenizedSentence]
    ) -> ArtifactProgram {
        let maximumSubwords = encoded.map(\.inputIds.count).max() ?? 0
        let maximumTokens = sentences.map(\.tokens.count).max() ?? 0
        for program in programs
        where program.shapes.subwordCount >= maximumSubwords
            && program.shapes.tokenCount >= maximumTokens
        {
            return program
        }
        return programs.last!
    }

    private func module(for program: ArtifactProgram) throws -> Module {
        if let module = modules[program.fileName] {
            return module
        }
        let programPath = artifact.directory
            .appendingPathComponent(program.fileName).path
        // Programs with separated data reference their weights in shared
        // .ptd files; every listed file must be loaded alongside.
        let dataFilePaths = (program.dataFiles ?? []).map {
            artifact.directory.appendingPathComponent($0).path
        }
        let module = dataFilePaths.isEmpty
            ? Module(filePath: programPath)
            : Module(filePath: programPath, dataFilePaths: dataFilePaths)
        try module.load()
        modules[program.fileName] = module
        return module
    }

    private func tagBatch(
        _ realSentences: [PretokenizedSentence],
        encoded realEncoded: [EncodedSentence],
        program: ArtifactProgram
    ) throws -> [TaggedSentence] {
        let shapes = program.shapes
        var sentences = realSentences
        var encoded = realEncoded
        while sentences.count < shapes.batchSize {
            sentences.append(sentences[sentences.count - 1])
            encoded.append(encoded[encoded.count - 1])
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

        let outputs = try module(for: program).forward(values)
        var probabilities: [[Float]] = []
        for output in outputs {
            guard let tensor: Tensor<Float> = output.tensor() else {
                throw PrismError.invalidArtifact("Program output is not a tensor.")
            }
            probabilities.append(try tensor.scalars())
        }
        return try decode(
            realSentences,
            probabilities: probabilities,
            tokenCount: shapes.tokenCount
        )
    }

    private func decode(
        _ sentences: [PretokenizedSentence],
        probabilities: [[Float]],
        tokenCount: Int
    ) throws -> [TaggedSentence] {
        let schema = artifact.labels.schema
        let uposCount = schema.upos.labels.count
        let ruleCount = schema.lemmaRules.rules.count

        var tagged: [TaggedSentence] = []
        for (row, sentence) in sentences.enumerated() {
            var tokens: [TaggedToken] = []
            for tokenIndex in 0..<sentence.tokens.count {
                let flat = row * tokenCount + tokenIndex

                let (uposIndex, uposConfidence) = argmax(
                    probabilities[0], offset: flat * uposCount, count: uposCount
                )

                var features: [String: [String]] = [:]
                var featureConfidences: [String: Double] = [:]
                for (featureIndex, feature) in schema.morphology.features.enumerated() {
                    let output = probabilities[1 + featureIndex]
                    if feature.allowsMultipleValues {
                        let count = feature.values.count
                        var selected: [String] = []
                        var confidence = Float.greatestFiniteMagnitude
                        for (valueIndex, value) in feature.values.enumerated() {
                            let probability = output[flat * count + valueIndex]
                            if probability > 0.5 {
                                selected.append(value)
                                confidence = min(confidence, probability)
                            }
                        }
                        if !selected.isEmpty {
                            features[feature.name] = selected
                            featureConfidences[feature.name] = Double(confidence)
                        }
                    } else {
                        let count = feature.values.count + 1
                        let (valueIndex, confidence) = argmax(
                            output, offset: flat * count, count: count
                        )
                        if valueIndex > 0 {
                            features[feature.name] = [feature.values[valueIndex - 1]]
                            featureConfidences[feature.name] = confidence
                        }
                    }
                }

                let (ruleIndex, lemmaConfidence) = argmax(
                    probabilities[probabilities.count - 1],
                    offset: flat * ruleCount,
                    count: ruleCount
                )
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

    private func argmax(_ values: [Float], offset: Int, count: Int) -> (Int, Double) {
        var bestIndex = 0
        var bestValue = -Float.infinity
        for index in 0..<count where values[offset + index] > bestValue {
            bestValue = values[offset + index]
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
