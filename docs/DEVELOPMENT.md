# Developing Prism

Setup, training, evaluation, and testing for working on Prism itself.
Application integration lives in [INTEGRATION.md](INTEGRATION.md); the
design rationale in [ARCHITECTURE.md](ARCHITECTURE.md); every accepted
decision and milestone in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## What the codebase provides today

Deterministic CoNLL-U ingestion; language-owned UPOS, morphology, and
lemma-rule schemas; the compact NorBERT4-xsmall student with shared
UPOS, per-feature morphology, and lemma edit-rule heads plus the
selected character-CNN branch and post-fusion morphology MLP;
schema-driven categorical and multi-label objectives with class-weighted
training; typed task-specific distillation (independent temperatures and
weights, optional decoupled knowledge distillation); supervised Apple
MPS training with reproducible checkpoints; exact, per-label, and
Rare/OOV development metrics; the licensed silver-data pipeline
(preparation, offline teacher labeling, training integration); per-head
temperature calibration; the versioned ExecuTorch export with parity
fixtures, program-data separation, multi-shape programs, and int8
quantization; and explicit language profiles so another language can
replace tokenizer, backbone, schemas, decoding policy, and artifact
metadata without touching shared code. The full decision history sits
in [PROJECT_STATUS.md](PROJECT_STATUS.md).

Not yet covered (explicit later tasks rather than hidden parts of the
tagger): dependency parsing, named entities, phrases, and multiword
expressions.

## Requirements

- macOS or another Python-compatible platform
- Python 3.12 (broadest compatibility with the current PyTorch and
  export ecosystem)
- Git
- For the native bindings: CMake ≥ 3.24 and a C++20 toolchain; Xcode
  with a current Swift toolchain for PrismKit; a JDK 21 for the Java
  binding

## Setup

Clone the repository, create the local virtual environment, and install
the development dependencies:

```bash
git clone git@github.com:dmlux/Prism.git
cd Prism
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev]'
```

The distribution is named `prism-nlp`; the Python import package
remains `prism`.

## Training data

Prism uses separately pinned official Universal Dependencies
repositories. They live under the ignored `data/raw/` directory and are
never committed to this repository.

```bash
git clone https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal.git \
  data/raw/UD_Norwegian-Bokmaal
git -C data/raw/UD_Norwegian-Bokmaal checkout \
  396d11f0c2bd290a2a2711015c04ac25bc3dcc06

git clone https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk.git \
  data/raw/UD_Norwegian-Nynorsk
git -C data/raw/UD_Norwegian-Nynorsk checkout \
  aaeb9d90c748c2bd9e272f180b599484f9f05ac6
```

Both datasets are distributed under CC BY-SA 4.0 and retain their own
attribution and share-alike requirements. Silver-corpus preparation
(NBdigital, municipal documents, Nynorsk Wikipedia) is documented in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

## Training, evaluation, and export

The complete pipeline — teacher, silver data, student distillation,
calibration, artifact export, and how to add a **new language** — is
documented step by step in [TRAINING.md](TRAINING.md). The commands
intentionally use training and development data only; the test split is
reserved for one final evaluation after the model and decision policy
are frozen, and the accepted results are recorded in
[benchmarks](benchmarks/).

## Tests

```bash
# Python (from the repository root)
python -m pytest python/tests
python -m ruff check python
python -m ruff format --check python
git diff --check

# C++ and Java (GoogleTest + the Java end-to-end run via ctest)
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure

# Swift (package manifest at the repository root)
swift test
```

The native suites validate against the local `models/` artifacts and
the shared parity fixtures; tests skip cleanly when local fixtures are
absent. The segmentation and source-mapping suites additionally use the
checked-in CC0 example texts under `data/examples/` (see the README
there), which are always present.

## Benchmarks

The reproducible performance suite for the C++ layer is built on
[Google Benchmark](https://github.com/google/benchmark) v1.9.1,
vendored and version-pinned under `cpp/vendor/benchmark/` like the
other third-party code, and built only when requested:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DPRISM_BENCHMARKS=ON
cmake --build cpp/build --target prism_benchmarks --parallel
cpp/build/prism_benchmarks
```

It measures, over the checked-in CC0 example texts: runtime
segmentation and subword BPE in isolation, tagger construction, and the
end-to-end variants `TagText` (raw text including segmentation) versus
`Tag(pretokenized)` (segmentation prepaid), each for the fp32 and the
fast (int8) artifact when present under `models/`. Document-scale runs
repeat the Bokmål text past 6,000 tokens to match the documented
document-inference protocol; throughput appears as `items_per_second`
(tokens/s). `PRISM_THREADS` overrides the CPU thread count for sweeps,
and the usual Google Benchmark flags apply (for example
`--benchmark_filter=TagText --benchmark_repetitions=5`).

For ad-hoc measurements of arbitrary texts and artifact variants, the
small `prism_chapter_benchmark` tool (cold/warm end-to-end run over a
text file) builds with the engine as before.

## PrismNative (Apple binary distribution)

`scripts/build-prism-native.sh` builds `PrismNative.xcframework` — the
Apple binary distribution of the stable C ABI (see
[INTEGRATION.md](INTEGRATION.md)) — from the regular CMake tree
(`-DPRISM_NATIVE=ON`), one self-contained dynamic library per slice:

```bash
scripts/build-prism-native.sh                       # all Apple slices
scripts/build-prism-native.sh --slices macos-arm64  # quick local slice
```

The third-party consumer proof lives in
`examples/prism-native-consumer` (plain C, no Prism-internal paths, no
linker flags) and runs against a locally built framework:

```bash
cd examples/prism-native-consumer
PRISM_NATIVE_XCFRAMEWORK_PATH=$PWD/../../build/prism-native/PrismNative.xcframework \
  swift run consumer ../../models/prism-no-0.2.2-fast
```

The `prism-native` workflow reproduces both in CI and attaches the
zipped XCFramework plus its SwiftPM checksum to `v*` releases.

## Repository layout

```text
Prism/
├── cpp/                 C++ API, C ABI, JNI bridge, vendored deps
│   ├── include/prism/   public headers (+ cpp/umbrella/prism)
│   ├── src/  tests/  tools/  benchmarks/  vendor/
├── docs/
│   ├── ARCHITECTURE.md  INTEGRATION.md  DEVELOPMENT.md
│   ├── PROJECT_STATUS.md  MODEL_STRATEGY.md
│   ├── BENCHMARKS.md    (model-development history)
│   └── benchmarks/      (one file per released artifact)
├── java/                Java API (Maven layout, pom.xml)
├── logos/
├── python/              training, evaluation, export, reference runtime
│   ├── src/prism/       data/ evaluation/ exporting/ languages/
│   │                    modeling/ schema/ training/
│   └── tests/
├── swift/               PrismKit sources (manifest: root Package.swift)
├── Prism.xcworkspace/   open this in Xcode
└── README.md
```

Datasets, checkpoints, virtual environments, caches, and generated
artifacts are excluded through `.gitignore`.
