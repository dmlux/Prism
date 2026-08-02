import Foundation

// The prebuilt ExecuTorch frameworks bundle the threadpool extension but
// do not surface it in the Objective-C API, so PrismKit binds the two
// exported functions directly (Itanium-mangled names; member functions
// take `this` as their first argument on this ABI).
@_silgen_name("_ZN10executorch9extension10threadpool14get_threadpoolEv")
private func executorchGetThreadpool() -> UnsafeMutableRawPointer?

@_silgen_name("_ZN10executorch9extension10threadpool10ThreadPool24_unsafe_reset_threadpoolEj")
private func executorchResetThreadpool(
    _ pool: UnsafeMutableRawPointer?,
    _ threadCount: UInt32
) -> Bool

/// CPU thread-count control for the ExecuTorch backend.
///
/// The runtime's own default parallelizes over every logical core, which
/// measurably oversubscribes Prism's small fixed-shape batches; the tagger
/// installs a measured default of six threads. Call ``setThreadCount(_:)``
/// before creating a tagger to choose explicitly.
public enum ComputeThreads {
    private static var overridden = false

    /// Overrides the CPU backend thread count for the whole process.
    /// Returns false when the pool cannot be resized.
    @discardableResult
    public static func setThreadCount(_ threadCount: Int) -> Bool {
        guard threadCount > 0, let pool = executorchGetThreadpool() else {
            return false
        }
        guard executorchResetThreadpool(pool, UInt32(threadCount)) else {
            return false
        }
        overridden = true
        return true
    }

    /// Installs a default without clobbering an explicit choice.
    static func installDefault(_ threadCount: Int) {
        guard !overridden,
            ProcessInfo.processInfo.activeProcessorCount > threadCount,
            let pool = executorchGetThreadpool()
        else {
            return
        }
        _ = executorchResetThreadpool(pool, UInt32(threadCount))
    }
}
