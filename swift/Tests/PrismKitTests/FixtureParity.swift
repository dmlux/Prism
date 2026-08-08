import ExecuTorch
import Foundation
import Testing

@testable import PrismKit

/// Shared engine-parity check, mirroring the C++ `ExpectFixtureParity`: run the
/// artifact's recorded fixture batch through the same engine and compare the
/// calibrated UPOS distribution against the recorded expectation within the
/// tolerance the artifact carries (tight for fp32, wider for the int8 twin).
enum FixtureParity {
    private enum Scalar: Decodable {
        case number(Double)
        case boolean(Bool)
        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let value = try? container.decode(Bool.self) {
                self = .boolean(value)
            } else {
                self = .number(try container.decode(Double.self))
            }
        }
        var int64: Int64 {
            switch self {
            case .number(let value): Int64(value)
            case .boolean(let value): value ? 1 : 0
            }
        }
        var bool: Bool {
            switch self {
            case .number(let value): value != 0
            case .boolean(let value): value
            }
        }
    }

    private struct FixtureInput: Decodable {
        let dtype: String
        let shape: [Int]
        let data: [Scalar]
    }
    private struct ExpectedOutput: Decodable { let data: [Double] }
    private struct Fixture: Decodable {
        let inputs: [FixtureInput]
        let expectedOutputs: [ExpectedOutput]
        enum CodingKeys: String, CodingKey {
            case inputs
            case expectedOutputs = "expected_outputs"
        }
    }
    private struct Comparison: Decodable {
        let probabilityTolerance: Double?
        enum CodingKeys: String, CodingKey {
            case probabilityTolerance = "probability_tolerance"
        }
    }
    private struct Fixtures: Decodable {
        let comparison: Comparison?
        let fixtures: [Fixture]
    }

    /// Whether a Prism artifact is present at `url` (its `manifest.json` exists).
    /// Callers gate parity tests on this via a `.enabled(if:)` trait, so the
    /// body below can assume the artifact is present.
    static func artifactExists(_ url: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: url.appendingPathComponent("manifest.json").path
        )
    }

    /// Asserts recorded parity for the artifact at `artifactURL`. Presence is
    /// guaranteed by the caller's `.enabled(if: artifactExists(_:))` trait.
    static func expect(artifactURL: URL) throws {
        let artifact = try PrismArtifact(contentsOf: artifactURL)
        let program = try artifact.program(for: .cpu)
        let dataFilePaths = (program.dataFiles ?? []).map {
            artifactURL.appendingPathComponent($0).path
        }
        let module = dataFilePaths.isEmpty
            ? Module(filePath: artifactURL.appendingPathComponent(program.fileName).path)
            : Module(
                filePath: artifactURL.appendingPathComponent(program.fileName).path,
                dataFilePaths: dataFilePaths
            )
        try module.load()

        let decoded = try JSONDecoder().decode(
            Fixtures.self,
            from: Data(contentsOf: artifactURL.appendingPathComponent("fixtures.json"))
        )
        let fixture = decoded.fixtures[0]
        // The tolerance travels with the artifact: fp32 fixtures record the
        // eager outputs (tight), int8 fixtures the quantized eager twin (wider).
        let tolerance = decoded.comparison?.probabilityTolerance ?? 5e-3

        var values: [Value] = []
        for input in fixture.inputs {
            switch input.dtype {
            case "int64":
                values.append(Value(Tensor<Int64>(input.data.map(\.int64), shape: input.shape)))
            case "bool":
                values.append(Value(Tensor<Bool>(input.data.map(\.bool), shape: input.shape)))
            default:
                Issue.record("Unexpected input dtype \(input.dtype)")
            }
        }

        let outputs = try module.forward(values)
        #expect(outputs.count == program.outputNames.count)
        // The lemma head is recorded as two top-k tensors, so the fixture holds
        // one expected entry more than the program's output list.
        #expect(outputs.count + 1 == fixture.expectedOutputs.count)

        // The first output is the calibrated UPOS distribution; compare it
        // against the recorded expectation at every position.
        let upos: Tensor<Float> = try #require(outputs[0].tensor())
        let produced = try upos.scalars()
        let expected = fixture.expectedOutputs[0].data
        #expect(produced.count == expected.count)
        var largestDifference = 0.0
        for index in produced.indices {
            largestDifference = max(largestDifference, abs(Double(produced[index]) - expected[index]))
        }
        #expect(largestDifference <= tolerance)

        // Every row must be a probability distribution over the UPOS labels.
        let labelCount = artifact.labels.schema.upos.labels.count
        let firstRow = produced.prefix(labelCount)
        #expect(abs(Double(firstRow.reduce(0, +)) - 1.0) <= 1e-3)
    }
}
