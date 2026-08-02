# Integrating Prism

Everything an application needs to ship Prism: the model-artifact
contract, and the integration details per language binding. For the
quick-start snippets, see the [README](../README.md); for the design
behind the pipeline, see [ARCHITECTURE.md](ARCHITECTURE.md).

## The model artifact

An artifact directory (for example `prism-no-0.2.2`) is the complete,
versioned integration contract:

- `model-xnnpack*.pte` — the lowered ExecuTorch programs, one per fixed
  shape. The production decoding policy is baked into the graph:
  morphology logit correction, per-head temperature calibration, and
  softmax/sigmoid. Programs therefore emit **final calibrated
  probabilities** (`*_probabilities` outputs, always float32).
  Consumers implement no decoding mathematics beyond argmax for
  exclusive heads and the 0.5 threshold for multi-valued morphology
  features.
- `model.ptd` — the shared tensor-data file (program-data separation).
  The weights live here exactly once; every program references them by
  content hash. A program's manifest entry lists its required data
  files under `data_files`; runtimes must load them alongside the
  program (all ExecuTorch runtimes accept data paths next to the
  program path).
- `vocabulary.json` — the complete Hugging Face fast-tokenizer
  definition (vocabulary, merges, normalization) of the subword
  tokenizer.
- `labels.json` — the label schema: UPOS labels, morphology features
  and values, lemma edit rules, and the character vocabulary with its
  maximum character count for the character-CNN inputs.
- `calibration.json`, `manifest.json`, `LICENSES/` — provenance: fitted
  temperatures, tensor contracts, shapes, checksums, and licensing.
- `fixtures.json` — recorded input/output parity fixtures. A
  development aid for validating an integration; do **not** ship it in
  app bundles.

Each model version comes in two precisions, and an application bundles
exactly one: the fp32 artifact is the exact reference behind the
published test benchmark; the **fast** artifact quantizes linears and
embeddings to int8 for less than half the size and up to twice the CPU
speed, with development-split quality within a few thousandths of a
percentage point of fp32 (per-version numbers in
[docs/benchmarks/](benchmarks/)). The Prism runtimes read either
artifact unchanged; the fast artifact requires the quantized kernel
library, which the C++ build links automatically and the Swift package
pulls in via the `kernels_quantized` product.

With several fixed-shape programs present, every Prism runtime sorts
sentences by length and runs each batch on the smallest program it fits
into — no configuration required.

### Tokenization: use the shipped definition or implement the contract

The model consumes pre-split words; turning words into subword IDs is
the integrator's side of the contract, with two supported routes:

1. **Use a Prism runtime or a ready-made engine.** The Prism bindings
   ship their own parity-tested segmentation and byte-level BPE.
   Alternatively, `vocabulary.json` is a standard `tokenizer.json`, so
   any Hugging Face tokenizers runtime loads it directly (for example
   [swift-transformers](https://github.com/huggingface/swift-transformers),
   `tokenizers` for Rust/Python, or the Java bindings).
2. **Implement the tokenizer yourself.** The manifest's tokenizer
   contract (file name, class name, padding token ID) together with
   `vocabulary.json` defines the exact behaviour; `fixtures.json`
   provides recorded inputs and outputs to verify the implementation
   token by token.

Character-CNN inputs are simpler: map each character of a word through
the `character_vocabulary` in `labels.json` and pad to the recorded
maximum count. Sentence splitting and word tokenization of raw text
remain the application's responsibility (or use the versioned
`prism-runtime-segmentation-v1` policy the Prism bindings implement);
applications that already have words can skip that layer entirely.

Input text is expected in Unicode NFC.

## Threads and compute device

Every Prism runtime installs a measured CPU thread-count default (the
ExecuTorch default parallelizes over all logical cores, which
oversubscribes the small fixed-shape batches). Override before loading
a tagger: `prism::engine::SetThreadCount` (C++),
`prism_set_thread_count` (C), `PrismTagger.setThreadCount` (Java),
`ComputeThreads.setThreadCount` (Swift).

Artifacts currently ship CPU (XNNPACK) programs only — a measured
decision: at the small batch sizes the runtimes use, GPU dispatch
overhead made MPS slower than CPU for this compact model, and CPU
programs run identically on Apple Silicon, Intel Macs, Windows, and
Linux. The device APIs (`automatic`/`cpu`/`gpu`) are already in place;
a future artifact with a GPU-lowered program lights them up without API
changes. `gpu` fails with a typed error on CPU-only artifacts;
`automatic` falls back to CPU.

Tagger instances are not thread-safe; results are immutable and freely
shareable.

## Swift (PrismKit)

`swift/PrismKit` implements the complete pipeline natively — its own
segmentation and byte-level BPE, parity-tested against the reference —
so no external tokenization framework is required.

- **Opening in Xcode:** open `Prism.xcworkspace` at the repository root
  (or `swift/Package.swift` directly) — Swift packages are native Xcode
  projects, so no generated `.xcodeproj` is committed.
- **Adding to an app:** depend on the package at `swift/` and on the
  ExecuTorch products `executorch`, `backend_xnnpack`,
  `kernels_optimized`, and `kernels_quantized`. Use a current
  `swiftpm-*` snapshot branch — the prebuilt frameworks must be
  compiled with a Swift toolchain your Xcode accepts — and keep the
  runtime at or above the exporter version that produced the artifact
  (the program format is backward compatible; the engine tests verify
  the pairing).
- **Force-loading backends:** ExecuTorch backends and kernels register
  through static initializers. App targets must add `-force_load` for
  the backend and kernel archives in Xcode's *Other Linker Flags* (see
  the ExecuTorch iOS documentation); the package's test target uses
  `-all_load` for the same reason.
- **Artifact placement:** ship the artifact directory as app resources
  (without `fixtures.json`).

## C++ and the C ABI

`cpp/` implements the same pipeline with hand-written scanners (no
regex engine, no ICU), parity-tested against the shared fixtures.

- **Building:** `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release`
  then `cmake --build cpp/build`. Third-party code is vendored and
  version-pinned under `cpp/vendor/`; only the ExecuTorch runtime is
  fetched at configure time, pinned to the exporter's exact version.
  `-DPRISM_ENGINE=OFF` builds segmentation and tokenization without
  network access.
- **Linking:** the CMake target `prism` aggregates every library and
  include directory: `target_link_libraries(app PRIVATE prism)`, then
  `#include <prism>`.
- **Torch headers:** the ExecuTorch build resolves torch headers
  through a Python interpreter; the repository virtual environment is
  wired as the default (`Python3_EXECUTABLE` overrides it).
- **Backend registration:** kernels and backends register through
  static initializers; the CMake setup already applies the required
  whole-archive linking to every consumer of `prism_tagger` and
  `prism_c`.
- **C ABI:** `<prism/prism_c.h>` exposes the tagger for applications
  whose core links plain C or crosses a foreign-function interface:
  opaque `prism_tagger`/`prism_result` handles, thread-local
  `prism_last_error()`, and accessors returning only C strings and
  scalars.

## Java

`java/` is a dependency-free Java 21 API (Kotlin-compatible out of the
box) over the native core: a thin JNI bridge marshals text in and one
flat payload out per call.

- **Building:** the canonical build produces `prism.jar` and the native
  library `prism_jni` through CMake (the Java binding is on by default
  when a JDK 21 is found; `-DPRISM_JAVA=OFF` disables it). Maven/Gradle
  consumers can alternatively build and install the JAR with
  `java/pom.xml` (`mvn install`, coordinates `io.github.dmlux:prism`).
- **Native library:** the JAR contains the Java layer only. At runtime
  the native library must be resolvable — either on
  `java.library.path` (`-Djava.library.path=...`) or loaded explicitly
  with `PrismTagger.loadNativeLibrary(path)` before the first `load`.

## Python

The Python package is the research and reference runtime: it runs the
frozen checkpoint eagerly through PyTorch (dynamic shapes, no
fixed-shape programs) and defines the behaviour every native binding is
parity-tested against. It requires the repository setup and a local
checkpoint rather than a released artifact — see
[DEVELOPMENT.md](DEVELOPMENT.md).
