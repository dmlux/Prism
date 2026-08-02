// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PrismKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v17),
    ],
    products: [
        .library(name: "PrismKit", targets: ["PrismKit"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/pytorch/executorch.git",
            branch: "swiftpm-1.4.0.20260731"
        )
    ],
    targets: [
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
)
