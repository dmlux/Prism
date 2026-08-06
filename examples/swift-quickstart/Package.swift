// swift-tools-version: 5.9
import PackageDescription

// Minimal PrismKit consumer: one executable, one dependency. The
// -all_load flag is PrismKit's documented registration contract — the
// prebuilt ExecuTorch backend and kernels register through static
// initializers the linker would otherwise drop (see
// docs/INTEGRATION.md in the Prism repository).
let package = Package(
    name: "swift-quickstart",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        // Plain version pinning — possible since 0.4.2, where Prism embeds
        // the prebuilt ExecuTorch frameworks as binary targets instead of
        // depending on upstream's snapshot branch. Apps that must never
        // float use .package(url: …, exact: "0.5.0") instead.
        .package(url: "https://github.com/dmlux/Prism.git", from: "0.5.0")
    ],
    targets: [
        .executableTarget(
            name: "quickstart",
            dependencies: [
                .product(name: "PrismKit", package: "Prism")
            ],
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-all_load"]),
                // -all_load pulls ExecuTorch's Apple image-processing
                // objects into the binary as well; they reference these
                // system frameworks (test bundles link them implicitly,
                // plain executables must be explicit).
                .linkedFramework("CoreImage"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreGraphics"),
            ]
        )
    ]
)
