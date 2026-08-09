import Foundation
import Testing

@testable import PrismKit

/// Executes each shipped program on its recorded fixture batch and verifies the
/// calibrated outputs against the values the exporter recorded — the same
/// recorded-parity contract the C++ `Engine` suite enforces, across the fp32
/// reference, the int8 fast twin, and the English ModernBERT artifact.
struct EngineParityTests {
    static var modelsRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("models")
    }

    @Test(.enabled(if: FixtureParity.artifactExists(
        EngineParityTests.modelsRoot.appendingPathComponent("prism-no-0.2.5")
    )))
    func executesNorwegianFixtureBatchWithRecordedParity() throws {
        try FixtureParity.expect(
            artifactURL: Self.modelsRoot.appendingPathComponent("prism-no-0.2.5")
        )
    }

    /// The fast artifact's fixtures record its quantized eager twin; parity
    /// against them validates the int8 program end to end.
    @Test(.enabled(if: FixtureParity.artifactExists(
        EngineParityTests.modelsRoot.appendingPathComponent("prism-no-0.2.5-fast")
    )))
    func executesFastArtifactFixturesWithRecordedParity() throws {
        try FixtureParity.expect(
            artifactURL: Self.modelsRoot.appendingPathComponent("prism-no-0.2.5-fast")
        )
    }

    /// The English artifact uses the ModernBERT/Ettin backbone; the
    /// language-independent runtime reproduces its recorded parity too.
    @Test(.enabled(if: FixtureParity.artifactExists(
        EngineParityTests.modelsRoot.appendingPathComponent("prism-en-0.1.0")
    )))
    func executesEnglishFixtureBatchWithRecordedParity() throws {
        try FixtureParity.expect(
            artifactURL: Self.modelsRoot.appendingPathComponent("prism-en-0.1.0")
        )
    }
}
