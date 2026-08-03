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

### Artifact metadata and language support

Whether a loaded artifact supports a document's language is decided
from the manifest, never guessed from directory or artifact names. All
bindings expose the same three `manifest.json` values, typed and
immutable: the artifact name, the artifact version, and the BCP 47
`language_tags` in manifest order (one artifact can support several —
the Norwegian artifact currently lists `nb` and `nn`). C++:
`Tagger::artifact()` / `prism::artifact::Artifact` (`name()`,
`version()`, `language_tags()`); C: `prism_tagger_artifact_name`,
`prism_tagger_artifact_version`, `prism_tagger_language_tag_count`,
`prism_tagger_language_tag`; Swift: `PrismTagger.artifactName`,
`.artifactVersion`, `.languageTags` (also on `ArtifactManifest`); Java:
`PrismTagger.artifactName()`, `.artifactVersion()`, `.languageTags()`.
Loading fails loudly when a manifest misses these fields.

## Source mapping

Applications that analyze raw text usually need to point back into the
text they passed in — to highlight a token, jump to a sentence, or
attach annotations. Every Prism runtime therefore returns, for every
token and sentence of a raw-text analysis, its origin in the exact,
unmodified input as **`Utf8ByteRange`** values.

The contract:

- A `Utf8ByteRange` is a **half-open `[start, end)` byte range in the
  UTF-8 encoding of the exact string the caller passed in**. Ranges are
  never empty, lie inside the input, sit on UTF-8 codepoint boundaries,
  and range lists are ordered and non-overlapping.
- The offsets always refer to the raw input — never to internally
  repaired, merged, whitespace-collapsed, or normalized intermediate
  strings. The mapping is carried *through* every transformation of the
  segmentation pipeline (restored spaces after sentence punctuation,
  merged line wraps, de-hyphenation, whitespace collapsing, chunking,
  batch sorting); it is never reconstructed afterwards by searching the
  input, so repeated identical tokens stay bound to their own
  occurrences.
- **A token has one or more ranges.** One range is the normal,
  contiguous case. A token assembled from several separated input
  fragments has one range per fragment: for the input
  `"språk-\nmodellen"` the model token stays `språkmodellen`, and its
  two ranges point exactly at `språk` and `modellen` — no invented
  range claims the removed `-\n` as token content. The model token text
  and the original bytes can therefore differ; the token text,
  `has_space_before`, and the source ranges are three distinct pieces
  of information, and none substitutes for another.
- **A sentence's ranges cover all its token fragments.** Fragments
  whose gap in the original text is pure whitespace share one range;
  gaps containing removed non-whitespace content (the `-` of a joined
  line wrap) split the sentence into several ranges. Chunks of an
  over-long sentence carry the parent's ranges clipped to their own
  tokens.
- **Pretokenized input has no source positions.** Without the raw text
  Prism cannot know any, and it never invents them: the range lists are
  simply empty (counts of 0 in the C ABI). Callers who own tokenization
  *and* offsets can pass their own validated ranges through the C++
  `PretokenizedSentence` fields or the Swift `PretokenizedSentence`
  initializer; they come back untouched on the results. The C and Java
  surfaces keep the simple pretokenized contract — such callers already
  hold their offsets, and Prism returns exactly one result per input
  token in order.

**UTF-8 byte offsets are not UTF-16 offsets.** Both encodings represent
the full Unicode repertoire, but they count different units: in `🙂å`,
the `å` starts at UTF-8 byte offset 4 (the emoji occupies four bytes)
yet at UTF-16 offset 2 (the emoji occupies two code units) — and `å`
itself occupies two UTF-8 bytes but one UTF-16 unit. Consumers whose
coordinate system is UTF-16 — Java string indices, `NSRange`-based
Apple APIs such as TextKit — must convert against the unchanged
original text; copying the numbers is not a conversion. The canonical
Prism representation stays `Utf8ByteRange` at the C++/C boundary; Swift
and Java additionally ship small, tested helpers against the original
string (`Utf8ByteRange.range(in:)` returning `Range<String.Index>`,
rejecting invalid boundaries; `Utf8ByteRange.utf16OffsetOf(text,
offset)` throwing on non-boundary offsets).

Input is still expected in NFC (the artifact's recorded normalization).
Prism does not normalize raw text internally, so the emitted offsets
always match the caller's actual bytes — visually identical but
differently encoded input (precomposed `å` versus `a` plus combining
ring) yields offsets into exactly the encoding that was passed in.

Per binding: C++ `TaggedToken::source_ranges` /
`TaggedSentence::source_ranges` (`<prism/utf8_byte_range.h>`); C
`prism_result_{token,sentence}_source_range_{count,start,end}`
(documented in `<prism/prism_c.h>`); Swift
`TaggedToken.sourceRanges` / `TaggedSentence.sourceRanges`; Java
`TaggedToken.sourceRanges()` / `TaggedSentence.sourceRanges()`. The
byte offsets are identical across all bindings for identical input;
shared test literals pin that parity.

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

**Prerequisites:** Xcode with a current Swift toolchain (Swift 5.9+;
the prebuilt ExecuTorch frameworks must have been compiled with a
toolchain your Xcode accepts — use a current `swiftpm-*` snapshot
branch), targeting macOS 13+ / iOS 17+. Nothing else: PrismKit has no
third-party Swift dependencies.

`swift/PrismKit` implements the complete pipeline natively — its own
segmentation and byte-level BPE, parity-tested against the reference —
so no external tokenization framework is required.

- **Opening in Xcode:** open `Prism.xcworkspace` at the repository root
  (or the root `Package.swift` directly) — Swift packages are native
  Xcode projects, so no generated `.xcodeproj` is committed.
- **Adding to an app:** depend on the Prism package (the manifest lives
  at the repository root so version pins resolve through the released
  `v*` tags, e.g. `.package(url: ..., from: "0.2.0")`) and on the
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

**Prerequisites:** CMake ≥ 3.24, a C++20 compiler (Apple Clang, Clang,
or GCC), and — only while *building* — a Python interpreter with
`torch` installed (the ExecuTorch build generates code through it; the
repository virtualenv from [DEVELOPMENT.md](DEVELOPMENT.md) is wired as
the default). Network access is needed once at configure time to fetch
the pinned ExecuTorch sources; `nlohmann/json` and GoogleTest are
vendored in-tree. The *built* libraries and your application have no
Python dependency.

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
- **Pinning a release:** consume the repository via FetchContent (or a
  submodule) pinned to a released `v*` tag:

  ```cmake
  FetchContent_Declare(prism
      GIT_REPOSITORY https://github.com/dmlux/Prism.git
      GIT_TAG v0.2.0
      SOURCE_SUBDIR cpp)
  FetchContent_MakeAvailable(prism)
  target_link_libraries(app PRIVATE prism)
  ```
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

**Prerequisites:** a JDK 21+ to build (the API uses records; any JRE
21+ suffices at runtime), plus everything from the C++ section — the
Java binding is a thin layer over the native core, so shipping Java
means building the C++ part once per target platform.

### How the JNI dependency works

The stack has exactly three pieces:

```text
your application (Java/Kotlin)
        │  classpath
   prism.jar          — pure Java, platform-independent, no dependencies
        │  System.loadLibrary("prism_jni")
libprism_jni.dylib/.so/prism_jni.dll
        │              — ONE self-contained native library per platform:
        │                the JNI bridge, the complete Prism C++ core,
        │                and the ExecuTorch runtime with all kernels,
        │                statically linked
model artifact directory (prism-no-…)
```

`prism.jar` contains only the Java layer (`PrismTagger`,
`TaggedSentence`, `TaggedToken`, `PrismException`) and declares
`native` methods. All actual work — segmentation, BPE, batching,
ExecuTorch execution, decoding — happens inside `prism_jni`, which
statically links the whole C++ core including the ExecuTorch runtime,
XNNPACK, and the optimized and quantized kernel libraries. There is
nothing else to install on the target machine: no ExecuTorch, no
PyTorch, no Python.

Design notes for anyone reimplementing or extending the bridge
(`cpp/src/jni.cpp`): text crosses into C++ as UTF-8 byte arrays
(`String.getBytes(UTF_8)`), because JNI's `GetStringUTFChars` uses
*modified* UTF-8, which corrupts supplementary codepoints; results
return as one flat parallel-array payload per call (a single JNI
transition regardless of sentence count), and strings travel back
through `NewString` (UTF-16) rather than `NewStringUTF`. The Java side
unflattens the payload into records.

### Building the native library

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
```

The Java binding is on by default when CMake finds a JDK 21 and JNI
headers (`-DPRISM_JAVA=OFF` disables it; `JAVA_HOME` steers which JDK
is found). This produces both deliverables:

- `cpp/build/prism.jar`
- `cpp/build/libprism_jni.dylib` (macOS) / `libprism_jni.so` (Linux) /
  `prism_jni.dll` (Windows)

The native library is platform- and architecture-specific: build it on
(or cross-compile for) every platform you ship — e.g. macOS arm64,
Linux x86_64, Windows x64. The JAR is the same everywhere.

### Wiring it into an application

Two resolution options at runtime:

```bash
# Option 1: library directory on java.library.path
java -Djava.library.path=/path/to/native/libs -cp prism.jar:app.jar my.App
```

```java
// Option 2: explicit path (e.g. after extracting from your own packaging)
PrismTagger.loadNativeLibrary(Path.of("/path/to/libprism_jni.dylib"));
try (PrismTagger tagger = PrismTagger.load(artifactDirectory)) { ... }
```

Maven/Gradle consumers normally just depend on the published
`io.github.dmlux:prism` from Maven Central (see below — it embeds the
common platform natives). For local development against unreleased
changes, build and install from `java/pom.xml` (`mvn install`); that
POM covers the Java layer, and the native library still comes from the
CMake build once per platform.

### Self-contained JARs (embedded natives)

There is no plugin mechanism that runs CMake on a *consumer's* machine
when they add a Maven/Gradle dependency — dependencies are downloaded
artifacts, and requiring every consumer to carry a C++ toolchain would
defeat the point. The established pattern (sqlite-jdbc, JNA, ONNX
Runtime) is the one Prism implements: **prebuilt natives inside the
JAR, extracted and loaded automatically at runtime.**

`PrismTagger` resolves the native library in this order:

1. an explicit `PrismTagger.loadNativeLibrary(path)` call;
2. a library embedded in the JAR under
   `/io/github/dmlux/prism/native/<os>-<arch>/` matching the current
   platform (`macos`/`linux`/`windows` × `aarch64`/`x86_64`), extracted
   to a temporary directory;
3. `System.loadLibrary("prism_jni")` via `java.library.path`.

To produce a self-contained JAR, place the platform builds under the
Maven resource path before packaging — each library is ~8 MB, so a JAR
covering several platforms stays reasonably sized:

```text
java/src/main/resources/io/github/dmlux/prism/native/
├── macos-aarch64/libprism_jni.dylib
├── macos-x86_64/libprism_jni.dylib
├── linux-x86_64/libprism_jni.so
└── windows-x86_64/prism_jni.dll
```

```bash
cmake --build cpp/build --parallel        # produces the current platform's library
mkdir -p java/src/main/resources/io/github/dmlux/prism/native/macos-aarch64
cp cpp/build/libprism_jni.dylib \
   java/src/main/resources/io/github/dmlux/prism/native/macos-aarch64/
cd java && mvn install                    # JAR now works without java.library.path
```

Consumers of such a JAR need nothing beyond the dependency and a model
artifact. This is exactly what the released JAR provides: CI builds the
natives for macOS (arm64, x86_64) and Linux (x86_64, aarch64) — one
runner per OS/architecture — and the combined JAR is published to
Maven Central as `io.github.dmlux:prism` and attached to the matching
GitHub `v*` release.

Common failure modes: `UnsatisfiedLinkError: no prism_jni in
java.library.path` — the library directory is not on the path and no
explicit load happened; `UnsatisfiedLinkError` naming a specific
method — `prism.jar` and the native library come from different builds
(rebuild both together); an immediate `PrismException` on `load` —
artifact directory missing or incomplete (`prism_last_error` details
travel into the exception message).

## Python

The Python package is the research and reference runtime: it runs the
frozen checkpoint eagerly through PyTorch (dynamic shapes, no
fixed-shape programs) and defines the behaviour every native binding is
parity-tested against. It requires the repository setup and a local
checkpoint rather than a released artifact — see
[DEVELOPMENT.md](DEVELOPMENT.md).
