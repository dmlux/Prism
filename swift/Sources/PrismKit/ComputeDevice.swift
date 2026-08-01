import Foundation

/// Where a Prism program executes.
///
/// The compute device is chosen when the model artifact is loaded: an
/// ExecuTorch program is lowered for one backend at export time, so the
/// artifact ships one program per supported backend and the runtime selects
/// among them. `cpu` maps to the XNNPACK program and works on every Mac,
/// including Intel machines built before Apple Silicon. `gpu` requires an
/// artifact that contains a GPU-lowered program and compatible hardware;
/// loading fails with ``PrismError/deviceUnavailable(_:)`` otherwise.
/// `automatic` picks the best available program and always succeeds when the
/// artifact contains a CPU program.
public enum ComputeDevice: String, Sendable, CaseIterable {
    case automatic
    case cpu
    case gpu
}

/// Errors surfaced by PrismKit's artifact loading and tagging APIs.
public enum PrismError: Error, Equatable {
    /// The artifact directory misses a required file.
    case missingArtifactFile(String)
    /// The artifact contains no program for the requested compute device.
    case deviceUnavailable(ComputeDevice)
    /// The artifact content contradicts its manifest contract.
    case invalidArtifact(String)
    /// A lemma edit rule removes more characters than the token contains.
    case invalidLemmaRule
}
