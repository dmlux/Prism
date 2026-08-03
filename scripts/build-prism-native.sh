#!/bin/bash
# Builds PrismNative.xcframework: the Apple binary distribution of the
# Prism C ABI (see docs/INTEGRATION.md, "PrismNative").
#
# One self-contained dynamic library per platform slice is built with the
# regular CMake tree (-DPRISM_NATIVE=ON), so the framework contains exactly
# the same C++ runtime sources as every other Prism build — no copies. The
# slices are then packaged with xcodebuild -create-xcframework together with
# the public C header and a module map, zipped deterministically, and the
# SwiftPM binary-target checksum is printed.
#
# Usage:
#   scripts/build-prism-native.sh [--slices macos-arm64,macos-x86_64,ios-arm64,ios-simulator-arm64,ios-simulator-x86_64]
#
# Slices of the same platform (macos-*, ios-simulator-*) are merged into one
# universal library with lipo, because an XCFramework carries exactly one
# library per platform variant.
#
# Requirements: Xcode (xcodebuild, iOS SDKs for the iOS slices), CMake,
# network access on the first configure (ExecuTorch FetchContent), and the
# repository Python environment for the ExecuTorch build.

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIRECTORY="$REPOSITORY_ROOT/build/prism-native"
SLICES="macos-arm64,macos-x86_64,ios-arm64,ios-simulator-arm64,ios-simulator-x86_64"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --slices)
        SLICES="$2"
        shift 2
        ;;
    *)
        echo "usage: $0 [--slices macos-arm64,ios-arm64,ios-simulator-arm64]" >&2
        exit 2
        ;;
    esac
done

# The iOS slices cross-compile through ExecuTorch's vendored
# ios.toolchain.cmake (the toolchain its own Apple builds use); a plain
# CMAKE_SYSTEM_NAME=iOS configure would leak the iOS deployment target into
# the host build of the flatc code generator. The toolchain file comes from
# the ExecuTorch checkout of an earlier slice, so build macos-arm64 first.
ios_toolchain_file() {
    for candidate in \
        "$OUTPUT_DIRECTORY/macos-arm64/_deps/executorch/third-party/ios-cmake/ios.toolchain.cmake" \
        "$REPOSITORY_ROOT/cpp/build/_deps/executorch/third-party/ios-cmake/ios.toolchain.cmake"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    echo "iOS slices need a fetched ExecuTorch checkout; build macos-arm64 first." >&2
    exit 2
}

cmake_flags_for_slice() {
    case "$1" in
    macos-arm64)
        echo "-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0"
        ;;
    macos-x86_64)
        # Cross-compiling needs CMAKE_SYSTEM_NAME to enter real cross mode,
        # otherwise ExecuTorch's architecture gates read the host CPU.
        echo "-DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
              -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0"
        ;;
    # DEPLOYMENT_TARGET stays 13.0 (not 17.0): ExecuTorch's flatc host-tool
    # build forwards the value as -mmacosx-version-min, and 13.0 is valid
    # for both OSes while 17.0 is no valid macOS version. A lower iOS
    # minimum in the library is harmless — the package still gates iOS 17+.
    ios-arm64)
        echo "-DCMAKE_TOOLCHAIN_FILE=$(ios_toolchain_file) \
              -DPLATFORM=OS64 -DDEPLOYMENT_TARGET=13.0"
        ;;
    ios-simulator-arm64)
        echo "-DCMAKE_TOOLCHAIN_FILE=$(ios_toolchain_file) \
              -DPLATFORM=SIMULATORARM64 -DDEPLOYMENT_TARGET=13.0"
        ;;
    ios-simulator-x86_64)
        echo "-DCMAKE_TOOLCHAIN_FILE=$(ios_toolchain_file) \
              -DPLATFORM=SIMULATOR64 -DDEPLOYMENT_TARGET=13.0"
        ;;
    *)
        echo "unknown slice: $1" >&2
        exit 2
        ;;
    esac
}

# 1. Build one self-contained libPrismNative.dylib per slice.
IFS=',' read -ra SLICE_LIST <<< "$SLICES"
for slice in "${SLICE_LIST[@]}"; do
    build_directory="$OUTPUT_DIRECTORY/$slice"
    echo "== building $slice"
    # shellcheck disable=SC2046
    cmake -S "$REPOSITORY_ROOT/cpp" -B "$build_directory" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPRISM_NATIVE=ON \
        -DPRISM_JAVA=OFF \
        $(cmake_flags_for_slice "$slice")
    cmake --build "$build_directory" --target prism_native --parallel
done

# 1b. Merge slices of the same platform into universal libraries — an
# XCFramework carries exactly one library per platform variant.
platform_of_slice() {
    case "$1" in
    macos-*) echo "macos" ;;
    ios-simulator-*) echo "ios-simulator" ;;
    ios-*) echo "ios-device" ;;
    esac
}

LIBRARY_ARGUMENTS=()
rm -rf "$OUTPUT_DIRECTORY/universal"
for platform in macos ios-device ios-simulator; do
    libraries=()
    for slice in "${SLICE_LIST[@]}"; do
        if [[ "$(platform_of_slice "$slice")" == "$platform" ]]; then
            libraries+=("$OUTPUT_DIRECTORY/$slice/libPrismNative.dylib")
        fi
    done
    [[ ${#libraries[@]} -gt 0 ]] || continue
    merged="$OUTPUT_DIRECTORY/universal/$platform/libPrismNative.dylib"
    mkdir -p "$(dirname "$merged")"
    if [[ ${#libraries[@]} -eq 1 ]]; then
        cp "${libraries[0]}" "$merged"
    else
        lipo -create "${libraries[@]}" -output "$merged"
    fi
    LIBRARY_ARGUMENTS+=(-library "$merged" -headers "$OUTPUT_DIRECTORY/include")
done

# 2. Stage the public headers: the stable C ABI plus a module map so the
# library is importable from Objective-C and Swift as well.
rm -rf "$OUTPUT_DIRECTORY/include"
mkdir -p "$OUTPUT_DIRECTORY/include/prism"
cp "$REPOSITORY_ROOT/cpp/include/prism/prism_c.h" "$OUTPUT_DIRECTORY/include/prism/"
cat > "$OUTPUT_DIRECTORY/include/module.modulemap" << 'EOF'
module PrismNative {
    header "prism/prism_c.h"
    export *
}
EOF

# 3. Assemble the XCFramework with licenses and notices.
rm -rf "$OUTPUT_DIRECTORY/PrismNative.xcframework" \
    "$OUTPUT_DIRECTORY/PrismNative.xcframework.zip"
xcodebuild -create-xcframework "${LIBRARY_ARGUMENTS[@]}" \
    -output "$OUTPUT_DIRECTORY/PrismNative.xcframework"
cp "$REPOSITORY_ROOT/LICENSE.md" "$OUTPUT_DIRECTORY/PrismNative.xcframework/"
cat > "$OUTPUT_DIRECTORY/PrismNative.xcframework/NOTICE.txt" << 'EOF'
PrismNative bundles, statically linked into every slice:
- Prism (Apache License 2.0, LICENSE.md)
- ExecuTorch, XNNPACK, cpuinfo, pthreadpool (BSD-style licenses,
  https://github.com/pytorch/executorch)
- nlohmann/json (MIT)
Model artifacts are separate deliverables with their own licenses.
EOF

# 4. Validate the result: every requested slice, only prism_* exports.
echo "== validating"
plutil -p "$OUTPUT_DIRECTORY/PrismNative.xcframework/Info.plist" | grep -E "LibraryIdentifier|SupportedPlatform" || true
for slice_directory in "$OUTPUT_DIRECTORY/PrismNative.xcframework"/*/; do
    library="$slice_directory/libPrismNative.dylib"
    [[ -f "$library" ]] || continue
    for arch in $(lipo -archs "$library"); do
        unexpected=$(nm -gU -arch "$arch" "$library" | awk '{print $3}' \
            | grep -cv '^_prism_' || true)
        exported=$(nm -gU -arch "$arch" "$library" | awk '{print $3}' \
            | grep -c '^_prism_' || true)
        if [[ "$unexpected" != "0" || "$exported" == "0" ]]; then
            echo "unexpected symbol surface in $library ($arch:" \
                "$exported prism_* exports, $unexpected foreign)" >&2
            exit 1
        fi
    done
done

# 5. Deterministic zip and the SwiftPM checksum.
(
    cd "$OUTPUT_DIRECTORY"
    find PrismNative.xcframework -exec touch -t 202001010000 {} +
    find PrismNative.xcframework | LC_ALL=C sort \
        | zip -X -q PrismNative.xcframework.zip -@
)
echo "== artifact"
ls -la "$OUTPUT_DIRECTORY/PrismNative.xcframework.zip"
echo "SwiftPM checksum:"
(cd "$REPOSITORY_ROOT" && swift package compute-checksum "$OUTPUT_DIRECTORY/PrismNative.xcframework.zip")
