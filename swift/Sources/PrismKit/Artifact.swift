import Foundation

/// Typed view of a versioned Prism model artifact directory.
///
/// The artifact is the cross-platform release contract: a manifest listing
/// one lowered program per backend, the label schema, the subword tokenizer
/// definition, and the calibration provenance. Programs emit final calibrated
/// probabilities, so consumers decode with argmax, the 0.5 threshold for
/// multi-valued morphology features, and ``LemmaEditRule/apply(to:)``.
public struct PrismArtifact: Sendable {
    public let directory: URL
    public let manifest: ArtifactManifest
    public let labels: ArtifactLabels

    public init(contentsOf directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard let manifestData = try? Data(contentsOf: manifestURL) else {
            throw PrismError.missingArtifactFile("manifest.json")
        }
        let manifest = try decoder.decode(ArtifactManifest.self, from: manifestData)

        let labelsURL = directory.appendingPathComponent(manifest.labelsFile)
        guard let labelsData = try? Data(contentsOf: labelsURL) else {
            throw PrismError.missingArtifactFile(manifest.labelsFile)
        }
        let labels = try decoder.decode(ArtifactLabels.self, from: labelsData)

        self.directory = directory
        self.manifest = manifest
        self.labels = labels
    }

    /// The lowered program serving the requested compute device.
    ///
    /// `cpu` selects the XNNPACK program, `gpu` a CoreML or MPS program, and
    /// `automatic` prefers GPU when the artifact provides one and falls back
    /// to CPU otherwise.
    public func program(for device: ComputeDevice) throws -> ArtifactProgram {
        guard let largest = try programs(for: device).last else {
            throw PrismError.deviceUnavailable(device)
        }
        return largest
    }

    /// All programs serving the device, sorted by capacity (smallest first).
    ///
    /// Artifacts may ship several fixed-shape programs per backend; runtimes
    /// pick the smallest program a batch fits into, so short sentences never
    /// pay the padding cost of the largest shapes.
    public func programs(for device: ComputeDevice) throws -> [ArtifactProgram] {
        let gpuBackends: Set<String> = ["coreml", "mps"]
        let backendPrograms: [ArtifactProgram]
        switch device {
        case .cpu:
            backendPrograms = manifest.programs.filter { $0.backend == "xnnpack" }
        case .gpu:
            backendPrograms = manifest.programs.filter {
                gpuBackends.contains($0.backend)
            }
        case .automatic:
            if let programs = try? programs(for: .gpu), !programs.isEmpty {
                return programs
            }
            backendPrograms = manifest.programs.filter { $0.backend == "xnnpack" }
        }
        guard !backendPrograms.isEmpty else {
            throw PrismError.deviceUnavailable(device)
        }
        return backendPrograms.sorted {
            ($0.shapes.subwordCount, $0.shapes.tokenCount)
                < ($1.shapes.subwordCount, $1.shapes.tokenCount)
        }
    }
}

public struct ArtifactManifest: Decodable, Sendable {
    public let manifestFormatVersion: Int
    public let artifactName: String
    public let artifactVersion: String
    public let languageTags: [String]
    public let labelsFile: String
    public let vocabularyFile: String
    public let characterUnicodeNormalization: String
    public let tokenizer: TokenizerContract
    public let programs: [ArtifactProgram]
    /// Shared external tensor-data files (program-data separation); absent
    /// in artifacts whose programs embed their weights.
    public let dataFiles: [ArtifactDataFile]?
    public let calibrationFile: String?
}

public struct ArtifactDataFile: Decodable, Sendable {
    public let fileName: String
    public let sha256: String
    public let sizeBytes: Int
}

public struct TokenizerContract: Decodable, Sendable {
    public let fileName: String
    public let className: String
    public let paddingTokenId: Int
}

public struct ArtifactProgram: Decodable, Sendable {
    public let fileName: String
    public let format: String
    public let backend: String
    public let precision: String
    public let sha256: String
    public let sizeBytes: Int
    public let shapes: FixedShapes
    public let outputNames: [String]
    /// External tensor-data files required at load time; nil or empty when
    /// the weights live inside the program file.
    public let dataFiles: [String]?
}

public struct FixedShapes: Decodable, Sendable {
    public let batchSize: Int
    public let subwordCount: Int
    public let tokenCount: Int
    public let characterCount: Int?
}

public struct ArtifactLabels: Decodable, Sendable {
    public let labelsFormatVersion: Int
    public let schema: LabelSchema
    public let characterVocabulary: CharacterVocabulary?
    public let maximumCharacterCount: Int?
}

public struct LabelSchema: Decodable, Sendable {
    public let upos: UposSchema
    public let morphology: MorphologySchema
    public let lemmaRules: LemmaRuleSchema
}

public struct UposSchema: Decodable, Sendable {
    public let labels: [String]
}

public struct MorphologySchema: Decodable, Sendable {
    public let features: [MorphologyFeature]
}

public struct MorphologyFeature: Decodable, Sendable {
    public let name: String
    public let values: [String]
    public let allowsMultipleValues: Bool
}

public struct LemmaRuleSchema: Decodable, Sendable {
    public let rules: [LemmaEditRule]
}

/// A lemma as a reversible edit of the token form.
///
/// Mirrors the Python reference: removals and additions operate on
/// characters of the NFC-normalized token.
public struct LemmaEditRule: Decodable, Sendable, Equatable {
    public let prefixRemoval: Int
    public let suffixRemoval: Int
    public let prefixAddition: String
    public let suffixAddition: String

    public func apply(to token: String) throws -> String {
        let characters = Array(token)
        guard prefixRemoval + suffixRemoval <= characters.count else {
            throw PrismError.invalidLemmaRule
        }
        let unchanged = characters[prefixRemoval..<(characters.count - suffixRemoval)]
        return prefixAddition + String(unchanged) + suffixAddition
    }
}

public struct CharacterVocabulary: Decodable, Sendable {
    public let characters: [String]

    /// Character-to-ID lookup for the character-CNN inputs.
    public func identifiers() -> [String: Int] {
        var lookup: [String: Int] = [:]
        for (index, character) in characters.enumerated() {
            lookup[character] = index
        }
        return lookup
    }
}
