# Swift quickstart (PrismKit)

The smallest complete PrismKit consumer: a SwiftPM executable with one
dependency. Requires macOS 13+ and network access on first build
(SwiftPM resolves Prism and the prebuilt ExecuTorch frameworks).

```bash
# From this directory, with an unpacked model (see ../README.md):
swift run -c release quickstart ../../prism-no-0.2.2-fast
```

Expected output: the artifact identity line followed by one line per
token (`text  UPOS  lemma  confidence`).

Notes:

- `-all_load` in `Package.swift` is required: the ExecuTorch backend
  and kernels register through static initializers.
- In an app, ship the artifact folder as a bundle *folder reference*
  and resolve it from `Bundle.main.resourceURL` instead of a CLI
  argument.
- Intel Macs: PrismKit's prebuilt ExecuTorch frameworks are arm64-only;
  use the PrismNative product (`../prism-native-consumer/`) or the
  CMake path there.
