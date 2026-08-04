// swift-tools-version: 5.9
import Foundation
import PackageDescription

// PrismNative is the binary distribution of the stable Prism C ABI for
// Apple projects with a C/C++ core (see docs/INTEGRATION.md). The
// XCFramework is produced by scripts/build-prism-native.sh and attached to
// v* releases; the release automation fills in the URL and checksum below.
// For local development and the consumer test, point
// PRISM_NATIVE_XCFRAMEWORK_PATH at a locally built PrismNative.xcframework.
let prismNativeReleaseURL =
    "https://github.com/dmlux/Prism/releases/download/v0.4.1/PrismNative.xcframework.zip"
let prismNativeReleaseChecksum =
    "4409ce9ac70ed8056a66ba7b5426994960d34f4c95e3059b9e1c4b159e8dcbe6"
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
var targets: [Target] = [
    // The manifest lives at the repository root so SwiftPM consumers
    // can pin released versions through plain `v*` git tags; the
    // sources stay under swift/ next to the other language bindings.
    .target(
        name: "PrismKit",
        dependencies: [
            .product(name: "executorch", package: "executorch"),
            .product(name: "backend_xnnpack", package: "executorch"),
            .product(name: "kernels_optimized", package: "executorch"),
            // int8 (fast) artifacts fuse the embedding lookup into
            // quantized_decomposed ops; this library provides them.
            .product(name: "kernels_quantized", package: "executorch"),
        ],
        path: "swift/Sources/PrismKit"
    ),
    .testTarget(
        name: "PrismKitTests",
        dependencies: [
            "PrismKit",
            .product(name: "backend_xnnpack", package: "executorch"),
        ],
        path: "swift/Tests/PrismKitTests",
        resources: [.process("Resources")],
        linkerSettings: [
            .unsafeFlags(["-Xlinker", "-all_load"])
        ]
    ),
]

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
    dependencies: [
        .package(
            url: "https://github.com/pytorch/executorch.git",
            branch: "swiftpm-1.4.0.20260731"
        )
    ],
    targets: targets
)
