import ExecuTorch
import XCTest

@testable import PrismKit

/// The English artifact uses the ModernBERT/Ettin backbone with a different
/// tokenizer (null unk_token, "[CLS]"/"[SEP]" template, byte-level GPT-2
/// pre-tokenizer, plain NFC) and its own abbreviations, all read from the
/// artifact. PrismKit must reproduce the Python reference through the same
/// language-independent code paths it runs for Norwegian.
final class EnglishParityTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var artifactURL: URL {
        repositoryRoot.appendingPathComponent("models/prism-en-0.1.0")
    }

    /// End-to-end oracle: Swift segmentation with the artifact's own
    /// abbreviations plus the ModernBERT byte-level BPE must reproduce the
    /// Python pipeline's subword IDs for every fixture sentence.
    func testEnglishSubwordIdsMatchPythonReference() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: artifactURL.appendingPathComponent("manifest.json").path
            ),
            "Local English artifact is not present."
        )
        let artifact = try PrismArtifact(contentsOf: artifactURL)
        let abbreviations = try XCTUnwrap(artifact.manifest.segmentation?.abbreviations)
        let policy = SegmentationPolicy(
            abbreviationTokens: Set(abbreviations),
            maximumTokenCount: 128
        )
        let tokenizer = try SubwordTokenizer(
            vocabularyURL: artifactURL.appendingPathComponent("vocabulary.json")
        )

        struct Oracle: Decodable {
            let sentenceInputIds: [[Int]]
            enum CodingKeys: String, CodingKey {
                case sentenceInputIds = "sentence_input_ids"
            }
        }
        let text = try String(
            contentsOf: repositoryRoot.appendingPathComponent(
                "data/examples/harbor-english.txt"
            ),
            encoding: .utf8
        )
        let oracle = try JSONDecoder().decode(
            Oracle.self,
            from: Data(
                contentsOf: repositoryRoot.appendingPathComponent(
                    "data/examples/harbor-english-subword-parity.json"
                )
            )
        )

        let sentences = RuntimeSegmentation.segment(text, policy: policy)
        XCTAssertEqual(sentences.count, oracle.sentenceInputIds.count)
        for (index, sentence) in sentences.enumerated() {
            XCTAssertEqual(
                tokenizer.encode(sentence).inputIds,
                oracle.sentenceInputIds[index],
                "harbor-english sentence \(index): \(sentence.tokens)"
            )
        }
    }

    /// The English program loads and runs through the same engine, producing a
    /// calibrated probability distribution per token.
    func testEnglishEngineExecutesFixtureBatch() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: artifactURL.appendingPathComponent("manifest.json").path
            ),
            "Local English artifact is not present."
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
        let fixture = try JSONDecoder().decode(
            Fixtures.self,
            from: Data(contentsOf: artifactURL.appendingPathComponent("fixtures.json"))
        ).fixtures[0]

        var values: [Value] = []
        for input in fixture.inputs {
            switch input.dtype {
            case "int64":
                values.append(Value(Tensor<Int64>(input.data.map(\.int64), shape: input.shape)))
            case "bool":
                values.append(Value(Tensor<Bool>(input.data.map(\.bool), shape: input.shape)))
            default:
                XCTFail("Unexpected input dtype \(input.dtype)")
            }
        }

        let outputs = try module.forward(values)
        XCTAssertEqual(outputs.count, program.outputNames.count)
        let upos: Tensor<Float> = try XCTUnwrap(outputs[0].tensor())
        let labelCount = artifact.labels.schema.upos.labels.count
        XCTAssertEqual(
            upos.shape,
            [program.shapes.batchSize, program.shapes.tokenCount, labelCount]
        )
        let probabilities = try upos.scalars()
        XCTAssertTrue(probabilities.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        let firstRow = probabilities.prefix(labelCount)
        XCTAssertEqual(firstRow.reduce(0, +), 1.0, accuracy: 1e-3)
    }
}
