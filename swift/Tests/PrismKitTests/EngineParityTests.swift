import XCTest

@testable import PrismKit

/// Executes each shipped program on its recorded fixture batch and verifies the
/// calibrated outputs against the values the exporter recorded — the same
/// recorded-parity contract the C++ `Engine` suite enforces, across the fp32
/// reference, the int8 fast twin, and the English ModernBERT artifact.
final class EngineParityTests: XCTestCase {
    private var modelsRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models")
    }

    func testExecutesNorwegianFixtureBatchWithRecordedParity() throws {
        try FixtureParity.expect(artifactURL: modelsRoot.appendingPathComponent("prism-no-0.2.3"))
    }

    /// The fast artifact's fixtures record its quantized eager twin; parity
    /// against them validates the int8 program end to end.
    func testExecutesFastArtifactFixturesWithRecordedParity() throws {
        try FixtureParity.expect(
            artifactURL: modelsRoot.appendingPathComponent("prism-no-0.2.3-fast")
        )
    }

    /// The English artifact uses the ModernBERT/Ettin backbone; the
    /// language-independent runtime reproduces its recorded parity too.
    func testExecutesEnglishFixtureBatchWithRecordedParity() throws {
        try FixtureParity.expect(artifactURL: modelsRoot.appendingPathComponent("prism-en-0.1.0"))
    }
}
