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
            branch: "swiftpm-1.3.1"
        )
    ],
    targets: [
        .target(
            name: "PrismKit",
            dependencies: [
                .product(name: "executorch", package: "executorch"),
                .product(name: "backend_xnnpack", package: "executorch"),
                .product(name: "kernels_optimized", package: "executorch"),
            ]
        ),
        .testTarget(
            name: "PrismKitTests",
            dependencies: [
                "PrismKit",
                .product(name: "backend_xnnpack", package: "executorch"),
            ],
            resources: [.process("Resources")],
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-all_load"])
            ]
        ),
    ]
)
