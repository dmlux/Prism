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

# Swift
cd swift && swift test
```

The native suites validate against the local `models/` artifacts and
the shared parity fixtures; tests skip cleanly when local fixtures are
absent.

## Repository layout

```text
Prism/
├── cpp/                 C++ API, C ABI, JNI bridge, vendored deps
│   ├── include/prism/   public headers (+ cpp/umbrella/prism)
│   ├── src/  tests/  tools/  vendor/
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
├── swift/               PrismKit (SwiftPM package)
├── Prism.xcworkspace/   open this in Xcode
└── README.md
```

Datasets, checkpoints, virtual environments, caches, and generated
artifacts are excluded through `.gitignore`.
