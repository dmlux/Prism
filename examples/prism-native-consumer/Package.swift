// swift-tools-version: 5.9
import PackageDescription

// A deliberately minimal third-party consumer of the PrismNative binary
// product: a plain C executable that adds the Prism package, selects the
// PrismNative product, includes <prism/prism_c.h>, and uses the tagger —
// with no Prism-internal header paths and no manual linker flags. Run it
// from this directory with the locally built XCFramework:
//
//   PRISM_NATIVE_XCFRAMEWORK_PATH=$PWD/../../build/prism-native/PrismNative.xcframework \
//     swift run consumer ../../models/prism-no-0.2.4-fast
let package = Package(
    name: "prism-native-consumer",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "consumer",
            dependencies: [
                .product(name: "PrismNative", package: "prism")
            ]
        ),
        // Coexistence and result-parity proof: one process links PrismKit
        // (Swift, prebuilt ExecuTorch products) and PrismNative (binary C
        // ABI) side by side and compares their linguistic results. The
        // -all_load flag is PrismKit's documented registration contract;
        // PrismNative itself needs no flags. Run with PRISM_ARTIFACT set:
        //   PRISM_NATIVE_XCFRAMEWORK_PATH=… PRISM_ARTIFACT=… swift test
        .testTarget(
            name: "CoexistenceTests",
            dependencies: [
                .product(name: "PrismKit", package: "prism"),
                .product(name: "PrismNative", package: "prism"),
            ],
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-all_load"])
            ]
        ),
    ]
)
