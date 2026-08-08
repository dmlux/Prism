// swift-tools-version: 5.9
import Foundation
import PackageDescription

// ExecuTorch's prebuilt Apple frameworks, embedded as binary targets.
//
// Upstream publishes its SwiftPM manifest only on snapshot *branches*
// (swiftpm-<version>), never as version tags — and SwiftPM refuses branch
// dependencies inside version-resolved packages, which would make every
// `.package(url: …, from:/exact:)` consumer of Prism unresolvable. Prism
// therefore declares the same binary artifacts the snapshot manifest
// declares (identical URLs and checksums, release variants), keeping this
// manifest fully version-stable. When bumping ExecuTorch, copy the new
// checksums from Package.swift on the corresponding swiftpm-* branch and
// keep the version in sync with the CMake/C++ build.
let executorchVersion = "1.4.0.20260731"
let executorchURL = "https://ossci-ios.s3.amazonaws.com/executorch/"
let executorchChecksums = [
    "executorch": "d1d0ae8c7918655e0b0781b11f839186db8b6bce248ee8aab3c2ece62fe6b494",
    "backend_xnnpack": "a46556574b2f715bf5e77475023ddddf1fe32ae00c25b2304c6a7f17a0bee75c",
    "kernels_optimized": "ce9b904f9cb069ebca430033c475c78009923191a9f2ee73e7f0657f2f4f37e4",
    "kernels_quantized": "3b35d5b4c95a3c0e0e3cd79a7786af4a14b183ab88bf6ac197d5b81da0206e56",
    "threadpool": "c9a53643dcfea95e4d47978020645f3fddc97b95d06ae0a6b263dc1bdb9f8c5a",
]

// PrismNative is the binary distribution of the stable Prism C ABI for
// Apple projects with a C/C++ core (see docs/INTEGRATION.md). The
// XCFramework is produced by scripts/build-prism-native.sh and attached to
// v* releases; the release automation fills in the URL and checksum below.
// For local development and the consumer test, point
// PRISM_NATIVE_XCFRAMEWORK_PATH at a locally built PrismNative.xcframework.
let prismNativeReleaseURL =
    "https://github.com/dmlux/Prism/releases/download/v0.6.0/PrismNative.xcframework.zip"
let prismNativeReleaseChecksum =
    "f6f86519a65e20d3757e6ad06d1157e412e7d772da7a36f77238c9f2715fea5b"
// SwiftPM requires local binary-target paths to be relative to the package
// root, so absolute environment values are relativized here.
let prismNativeLocalPath: String? = {
    guard var path = ProcessInfo.processInfo.environment["PRISM_NATIVE_XCFRAMEWORK_PATH"]
    else { return nil }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path + "/"
    if path.hasPrefix(root) {
        path = String(path.dropFirst(root.count))
    }
    return path
}()

var products: [Product] = [
    .library(name: "PrismKit", targets: ["PrismKit"])
]
var targets: [Target] = executorchChecksums.map { name, checksum in
    .binaryTarget(
        name: name,
        url: "\(executorchURL)\(name)-\(executorchVersion).zip",
        checksum: checksum
    )
}
targets.append(contentsOf: [
    // The manifest lives at the repository root so SwiftPM consumers
    // can pin released versions through plain `v*` git tags; the
    // sources stay under swift/ next to the other language bindings.
    .target(
        name: "PrismKit",
        dependencies: [
            "executorch",
            // backend_xnnpack and kernels_optimized require the threadpool.
            "backend_xnnpack",
            "kernels_optimized",
            // int8 (fast) artifacts fuse the embedding lookup into
            // quantized_decomposed ops; this library provides them.
            "kernels_quantized",
            "threadpool",
        ],
        path: "swift/Sources/PrismKit",
        linkerSettings: [
            // What upstream's wrapper targets add for these products:
            // libc++ for the executorch runtime, Accelerate for the
            // optimized kernels.
            .linkedLibrary("c++"),
            .linkedFramework("Accelerate"),
        ]
    ),
    .testTarget(
        name: "PrismKitTests",
        dependencies: [
            "PrismKit"
        ],
        path: "swift/Tests/PrismKitTests",
        resources: [.process("Resources")],
        linkerSettings: [
            .unsafeFlags(["-Xlinker", "-all_load"])
        ]
    ),
])

if let localPath = prismNativeLocalPath {
    products.append(.library(name: "PrismNative", targets: ["PrismNative"]))
    targets.append(.binaryTarget(name: "PrismNative", path: localPath))
} else if !prismNativeReleaseURL.hasSuffix("PLACEHOLDER") {
    products.append(.library(name: "PrismNative", targets: ["PrismNative"]))
    targets.append(
        .binaryTarget(
            name: "PrismNative",
            url: prismNativeReleaseURL,
            checksum: prismNativeReleaseChecksum
        )
    )
}

let package = Package(
    name: "PrismKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v17),
    ],
    products: products,
    targets: targets
)
