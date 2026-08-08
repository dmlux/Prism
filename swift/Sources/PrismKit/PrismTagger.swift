import ExecuTorch
import Foundation

/// One tagged token with calibrated confidences per decision.
///
/// `sourceRanges` locates the token in the exact string passed to
/// ``PrismTagger/tag(text:)`` as ordered, non-overlapping ``Utf8ByteRange``
/// values — UTF-8 byte offsets into that string's UTF-8 view, not UTF-16
/// offsets. A token whose source is contiguous has exactly one range; a
/// token assembled from several separated input fragments (for example a
/// de-hyphenated line wrap, "språk-\nmodellen" → "språkmodellen") has one
/// range per contributing fragment. `text` may differ from the bytes the
/// ranges point to after internal repairs — `text`, `hasSpaceBefore`, and
/// `sourceRanges` are three distinct pieces of information. The list is
/// empty for pretokenized input without caller-supplied ranges.
public struct TaggedToken: Sendable {
    public let text: String
    public let hasSpaceBefore: Bool
    public let upos: String
    public let uposConfidence: Double
    public let features: [String: [String]]
    public let featureConfidences: [String: Double]
    public let lemma: String
    public let lemmaConfidence: Double
    public let sourceRanges: [Utf8ByteRange]

    /// The complete calibrated UPOS probability distribution of this token:
    /// one entry per label of the loaded artifact, sorted by descending
    /// probability (the first entry is the decision reported by ``upos``
    /// and ``uposConfidence``), summing to ~1.
    public let uposDistribution: [UposProbability]
}

/// One entry of a token's UPOS probability distribution.
public struct UposProbability: Sendable, Equatable {
    public let upos: String
    public let probability: Double
}

/// One tagged sentence in original token order.
///
/// `sourceRanges` covers every token fragment of the sentence in the exact
/// raw-text input: fragments whose gap in the original is pure whitespace
/// share one range, gaps containing removed non-whitespace content split
/// the sentence into several ranges. Empty for pretokenized input without
/// caller-supplied ranges.
public struct TaggedSentence: Sendable {
    public let tokens: [TaggedToken]
    public let sourceRanges: [Utf8ByteRange]
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
        // The segmentation inventory travels in the artifact so every language
        // segments with its own abbreviations; an empty one is a hard error
        // rather than a silent fallback that would mis-segment the language.
        guard let abbreviations = artifact.manifest.segmentation?.abbreviations,
            !abbreviations.isEmpty
        else {
            throw PrismError.invalidArtifact(
                "Artifact manifest declares no segmentation abbreviations; the "
                    + "runtime cannot segment without the model's inventory."
            )
        }
        segmentationPolicy = SegmentationPolicy(
            abbreviationTokens: Set(abbreviations),
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

    /// The artifact name recorded in the loaded manifest (for example
    /// "prism-no").
    public var artifactName: String { artifact.manifest.artifactName }

    /// The artifact version recorded in the loaded manifest.
    public var artifactVersion: String { artifact.manifest.artifactVersion }

    /// The BCP 47 language tags the loaded artifact supports (currently for
    /// example "nb" and "nn"), in manifest order. Decide language support
    /// from these values, never from directory or artifact names.
    public var languageTags: [String] { artifact.manifest.languageTags }

    /// Every UPOS tag the loaded artifact can assign, mirrored from its
    /// label schema (labels.json). Inventories differ per language
    /// artifact.
    public var uposLabels: [String] { artifact.labels.schema.upos.labels }

    /// Every morphology feature the loaded artifact can predict, with its
    /// possible values, in schema order. Inventories differ per language
    /// artifact.
    public var morphologyFeatures: [MorphologyFeature] {
        artifact.labels.schema.morphology.features
    }

    /// Segment raw text with the runtime policy, then tag every sentence.
    ///
    /// Every result carries ``Utf8ByteRange`` source ranges against the
    /// exact `text` argument's UTF-8 view.
    public func tag(text: String) throws -> [TaggedSentence] {
        try tag(sentences: RuntimeSegmentation.segment(text, policy: segmentationPolicy))
    }

    /// Tag application-supplied word tokens (space assumed between words).
    ///
    /// Without raw text there are no source positions: the results carry
    /// empty ``Utf8ByteRange`` lists, which Prism never invents.
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

    /// Tag sentences carrying their own spacing information.
    ///
    /// Callers who own tokenization and source offsets may fill the
    /// sentences' source-range fields (``PretokenizedSentence`` validates
    /// the invariants); the ranges travel through chunking and batching
    /// untouched and reappear on the corresponding results.
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
                var uposDistribution: [UposProbability] = []
                uposDistribution.reserveCapacity(uposCount)
                for label in 0..<uposCount {
                    uposDistribution.append(UposProbability(
                        upos: schema.upos.labels[label],
                        probability: Double(probabilities[0][flat * uposCount + label])
                    ))
                }
                uposDistribution.sort { $0.probability > $1.probability }

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
                        lemmaConfidence: lemmaConfidence,
                        sourceRanges: sentence.tokenSourceRanges.isEmpty
                            ? []
                            : sentence.tokenSourceRanges[tokenIndex],
                        uposDistribution: uposDistribution
                    )
                )
            }
            tagged.append(TaggedSentence(tokens: tokens, sourceRanges: sentence.sourceRanges))
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
