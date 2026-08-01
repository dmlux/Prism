import ExecuTorch
import XCTest

@testable import PrismKit

final class EngineSpikeTests: XCTestCase {
    func testExecutesFixtureBatchOnCpu() throws {
        let artifactURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models/prism-no-0.2.1")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: artifactURL.appendingPathComponent("manifest.json").path
            ),
            "Local artifact is not present."
        )

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

        struct FixtureInput: Decodable {
            let name: String
            let dtype: String
            let shape: [Int]
            let data: [Scalar]
        }
        enum Scalar: Decodable {
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
        struct Fixture: Decodable { let inputs: [FixtureInput] }
        struct Fixtures: Decodable { let fixtures: [Fixture] }
        let fixturesData = try Data(
            contentsOf: artifactURL.appendingPathComponent("fixtures.json")
        )
        let fixture = try JSONDecoder()
            .decode(Fixtures.self, from: fixturesData).fixtures[0]

        var values: [Value] = []
        for input in fixture.inputs {
            switch input.dtype {
            case "int64":
                let scalars = input.data.map(\.int64)
                values.append(Value(Tensor<Int64>(scalars, shape: input.shape)))
            case "bool":
                let scalars = input.data.map(\.bool)
                values.append(Value(Tensor<Bool>(scalars, shape: input.shape)))
            default:
                XCTFail("Unexpected input dtype \(input.dtype)")
            }
        }

        let outputs = try module.forward(values)

        XCTAssertEqual(outputs.count, program.outputNames.count)
        let upos: Tensor<Float> = try XCTUnwrap(outputs[0].tensor())
        XCTAssertEqual(
            upos.shape,
            [program.shapes.batchSize, program.shapes.tokenCount, artifact.labels.schema.upos.labels.count]
        )
        let probabilities = try upos.scalars()
        XCTAssertTrue(probabilities.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        // Row of the first valid token must be a probability distribution.
        let firstRow = probabilities.prefix(artifact.labels.schema.upos.labels.count)
        XCTAssertEqual(firstRow.reduce(0, +), 1.0, accuracy: 1e-3)
    }
}
