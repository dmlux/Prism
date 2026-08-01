<p align="center">
  <img src="logos/logo.svg" alt="Prism" width="420">
</p>

# Prism

Prism is a modular, open-source NLP toolkit for fast, local, and
privacy-friendly linguistic analysis. It is designed to provide
language-specific models through a unified, versioned API for tokenization,
sentence segmentation, part-of-speech tagging, lemmatization, and
morphological analysis across Python and native platforms.

The current implementation focuses on Norwegian. Bokmål (`nb`) and Nynorsk
(`nn`) share one measured gold-only student over a common Norwegian schema,
while retaining separate data profiles and quality reports. Prism's shared
model, training, evaluation, export, and artifact contracts remain
language-independent.

## Current status

Prism currently provides:

- deterministic CoNLL-U ingestion and typed externally-tokenized inputs;
- language-owned UPOS, morphology, and lemma-rule schemas;
- a compact NorBERT4-xsmall Transformer student for Norwegian;
- shared UPOS, per-feature morphology, and lemma edit-rule heads;
- checkpoint-compatible linear and shared-MLP head architectures plus the
  selected structured, character-aware morphology/lemma path;
- schema-driven categorical and multi-label morphology objectives;
- supervised Apple MPS training with reproducible checkpoints;
- class-weighted morphology training derived only from the training split;
- exact, per-label, and threshold-independent development metrics;
- reproducible Rare/OOV development slices derived only from training-form
  frequencies;
- a selected, export-tested character-CNN branch that feeds complete-token form
  information into morphology and lemma while leaving UPOS unchanged;
- a selected shared post-fusion morphology MLP that improves complete UFeats
  and Gender across Bokmål and Nynorsk while preserving an explicit identity
  control for compatibility and ablations;
- a typed task-specific distillation policy with independent UPOS, morphology,
  and lemma temperatures and weights;
- optional categorical DKD with independently weighted target-class and
  non-target-class knowledge while preserving binary KL for multi-value tasks;
- a versioned ExecuTorch export spike with PyTorch parity coverage;
- explicit language profiles so another language can replace its tokenizer,
  backbone, schemas, decoding policy, and artifact metadata.

The selected twelve-epoch Mean-pooling, learned-last-four, wide-shared-MLP
Student combines a structured, soft-decision morphology refinement with a
compact character CNN for morphology and lemma, then distills an accepted
NorBERT4-Base Teacher at temperature 1.0 and weight 0.1. The character branch
improves Rare end-to-end lemma accuracy by 2.67/2.42 percentage points and Rare
morphology micro F1 by 1.76/1.50 points on Bokmål/Nynorsk. OOV lemma,
morphology, and UPOS also improve on both written standards. The selected
shared morphology MLP then corrects 226 additional complete UFeats bundles and
204 Gender decisions across Bokmål and Nynorsk for a 70,661,786-byte
checkpoint that remains below the 100 MB target. Decoupled knowledge distillation
further lowers loss and improves overall UPOS, lemma, and Rare/OOV lemma and
morphology on both standards without changing inference cost. Its localized
Rare/OOV UPOS tradeoffs remain explicit. Both official test splits remain
untouched.
See [the benchmark notes](docs/benchmarks.md) for the complete, comparable
results.

Prism does not yet ship a production model bundle or a stable native runtime
API. Dependency parsing, raw-text tokenization, sentence segmentation, named
entities, phrases, and multiword expressions are explicit later tasks rather
than hidden parts of the current token tagger.

## Architecture

The compact student converts externally supplied tokens into subwords, runs a
replaceable language backbone, aligns contextual subword states back to source
tokens, and applies shared task-head implementations:

```text
tokens + spacing
      |
language tokenizer
      |
subword IDs + alignment
      |
language backbone
      |
token representations
      +--> UPOS head
      +--> morphology feature heads
      +--> lemma edit-rule head
```

The Norwegian profile currently selects NorBERT4-xsmall. Generic Prism code
does not import or branch on NorBERT4. Future language profiles may select a
different tokenizer and backbone while reusing the same model and artifact
contracts.

For a learning-oriented explanation of the complete teacher-student and
native-inference design, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The accepted roadmap and licensing decisions are in
[docs/model-strategy.md](docs/model-strategy.md).

## Using the exported model artifact

An exported artifact directory (for example `models/prism-no-0.2.0`) is the
complete integration contract for native runtimes:

- `model-xnnpack.pte` — the lowered ExecuTorch program. The production
  decoding policy is baked into the graph: morphology logit correction,
  per-head temperature calibration, and softmax/sigmoid. The program
  therefore emits **final calibrated probabilities** (`*_probabilities`
  outputs, always float32). Consumers implement no decoding mathematics
  beyond argmax for exclusive heads and the 0.5 threshold for multi-valued
  morphology features.
- `vocabulary.json` — the complete Hugging Face fast-tokenizer definition
  (vocabulary, merges, normalization) of the subword tokenizer.
- `labels.json` — the label schema: UPOS labels, morphology features and
  values, lemma edit rules, and the character vocabulary with its maximum
  character count for the character-CNN inputs.
- `calibration.json`, `manifest.json`, `LICENSES/` — provenance: fitted
  temperatures, tensor contracts, shapes, checksums, and licensing.
- `fixtures.json` — recorded input/output parity fixtures. This file is a
  development aid for validating a native integration and is not part of
  the shipped bundle.

### Tokenization: use the shipped definition or implement the contract

The model consumes pre-split words; turning words into subword IDs is the
integrator's side of the contract, with two supported routes:

1. **Use a ready-made engine (recommended).** `vocabulary.json` is a
   standard `tokenizer.json`, so any Hugging Face tokenizers runtime loads
   it directly — for example
   [swift-transformers](https://github.com/huggingface/swift-transformers)
   on Apple platforms, `tokenizers` for Rust/Python, or the Java bindings.
   No tokenization rules have to be re-implemented.
2. **Implement the tokenizer yourself.** The manifest's tokenizer contract
   (file name, class name, padding token ID) together with
   `vocabulary.json` defines the exact behaviour; `fixtures.json` provides
   recorded inputs and outputs to verify the implementation token by token.

Character-CNN inputs are simpler: map each character of a word through the
`character_vocabulary` in `labels.json` and pad to the recorded maximum
count. Sentence splitting and word tokenization of raw text remain the
application's responsibility (or a port of the versioned
`prism-runtime-segmentation-v1` policy used by the Python API); applications
that already have words can skip that layer entirely.

## PrismKit (Swift)

`swift/PrismKit` is the native Swift implementation of the complete
pipeline. It ships its own word segmentation (the
`prism-runtime-segmentation-v1` policy) and its own byte-level BPE subword
tokenizer, both parity-tested against the Python reference, so no external
tokenization framework is required — maximum performance at equal quality
is the API's stated goal.

```swift
import PrismKit

let tagger = try PrismTagger(
    artifactURL: artifactDirectory,   // e.g. .../prism-no-0.2.0
    device: .cpu                      // .automatic, .cpu, .gpu
)
let sentences = try tagger.tag(text: "Hun kjøpte tre gamle bøker.")
// or: try tagger.tag(pretokenized: [["Hun", "kjøpte", "bøker", "."]])
for token in sentences[0].tokens {
    print(token.text, token.upos, token.lemma, token.uposConfidence)
}
```

Integration notes:

- **Opening in Xcode:** open `Prism.xcworkspace` at the repository root
  (or `swift/Package.swift` directly) — Swift packages are native Xcode
  projects, so no generated `.xcodeproj` is committed.
- **Adding to an app:** depend on the package at `swift/` and on the
  ExecuTorch products `executorch`, `backend_xnnpack`, and
  `kernels_optimized`. Use a current `swiftpm-*` snapshot branch — the
  prebuilt frameworks must be compiled with a Swift toolchain your Xcode
  accepts — and keep the runtime at or above the exporter version that
  produced the artifact (the program format is backward compatible; the
  test suite's engine tests verify the pairing).
- **Force-loading backends:** ExecuTorch backends register through static
  initializers. App targets must add `-force_load` for the backend and
  kernel archives in Xcode's *Other Linker Flags* (see the ExecuTorch iOS
  documentation); the package's test target uses `-all_load` for the same
  reason.
- **Compute device:** `.cpu` (XNNPACK) works on every Mac including
  Intel machines; `.gpu` requires an artifact containing a GPU-lowered
  program and fails with a typed `deviceUnavailable` error otherwise;
  `.automatic` picks the best available program.
- **Multi-shape artifacts:** when the artifact ships several fixed-shape
  programs, `PrismTagger` sorts sentences by length and runs every batch
  on the smallest program it fits into — no configuration required.
- **Artifact placement:** ship the artifact directory (program files plus
  `vocabulary.json`, `labels.json`, `calibration.json`, `manifest.json`)
  as app resources; `fixtures.json` is a development aid and does not
  belong in the bundle.

## C++ API and C ABI

`cpp/` is the native C++ implementation of the same pipeline — word
segmentation, byte-level BPE, ExecuTorch execution, and decoding — built
with hand-written scanners (no regex engine, no ICU) and parity-tested
against the same shared fixtures. Text is expected in Unicode NFC.

```cpp
#include <prism>  // umbrella header; or the individual <prism/*.h> headers

prism::tagger::Tagger tagger("models/prism-no-0.2.0");
const auto sentences = tagger.TagText("Hun kjøpte tre gamle bøker.");
for (const auto& token : sentences[0].tokens) {
    std::cout << token.text << " " << token.upos << " " << token.lemma
              << " " << token.upos_confidence << "\n";
}
```

For applications whose core links plain C (or crosses a foreign function
interface), `<prism/prism_c.h>` exposes the same functionality through a
stable C ABI: opaque `prism_tagger`/`prism_result` handles, thread-local
`prism_last_error()`, and accessors returning only C strings and scalars.

Integration notes:

- **Building:** `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release`
  then `cmake --build cpp/build`. Third-party code is vendored and
  version-pinned under `cpp/vendor/`; only the ExecuTorch runtime is
  fetched at configure time, pinned to the exporter's exact version.
  `-DPRISM_ENGINE=OFF` builds segmentation and tokenization without
  network access.
- **Linking:** the CMake target `prism` aggregates every library and
  both include directories, so consumers write
  `target_link_libraries(app PRIVATE prism)` and `#include <prism>`.
- **Torch headers:** the ExecuTorch build resolves torch headers through
  a Python interpreter; the repository virtual environment is wired as
  the default (`Python3_EXECUTABLE` overrides it).
- **Backend registration:** kernels and backends register through static
  initializers; the CMake setup already applies the required whole-archive
  linking to every consumer of `prism_tagger` and `prism_c`.
- **Multi-shape artifacts:** the tagger sorts sentences by length and
  runs every batch on the smallest fitting program — no configuration
  required.

## Java API

`java/` is a dependency-free Java 21 API (Kotlin-compatible out of the
box) over the native Prism core: a thin JNI bridge marshals text in and
results out, so segmentation, tokenization, batching, and decoding run
at native speed.

```java
try (PrismTagger tagger = PrismTagger.load(Path.of("models/prism-no-0.2.0"))) {
    for (TaggedSentence sentence : tagger.tagText("Hun kjøpte tre gamle bøker.")) {
        for (TaggedToken token : sentence.tokens()) {
            System.out.println(token.text() + " " + token.upos() + " "
                + token.lemma() + " " + token.uposConfidence());
        }
    }
}
```

Integration notes:

- **Building:** the repository's canonical build produces `prism.jar` and
  the native library `prism_jni` through CMake (see the C++ section; the
  Java binding is on by default when a JDK 21 is found, `-DPRISM_JAVA=OFF`
  disables it). Maven/Gradle consumers can alternatively build and
  install the JAR with `java/pom.xml` (`mvn install`, coordinates
  `io.github.dmlux:prism`).
- **Native library:** the JAR contains the Java layer only. At runtime
  the native library must be resolvable — either on `java.library.path`
  (`-Djava.library.path=...`) or loaded explicitly with
  `PrismTagger.loadNativeLibrary(path)` before the first `load`.
- **Threading:** a `PrismTagger` instance is not thread-safe; results are
  immutable and freely shareable.

## Requirements

- macOS or another Python-compatible platform
- Python 3.12
- Git

Python 3.12 is the supported development runtime because it has broad
compatibility with the current PyTorch and export ecosystem.

## Setup

Clone the repository, create the local virtual environment, and install the
development dependencies:

```bash
git clone git@github.com:dmlux/Prism.git
cd Prism
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev]'
```

The distribution is named `prism-nlp`; the Python import package remains
`prism`.

## Training data

Prism uses separately pinned official Universal Dependencies repositories.
They live under the ignored `data/raw/` directory and are never committed to
this repository.

### Norwegian Bokmål

```bash
git clone https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal.git \
  data/raw/UD_Norwegian-Bokmaal
git -C data/raw/UD_Norwegian-Bokmaal checkout \
  396d11f0c2bd290a2a2711015c04ac25bc3dcc06
```

### Norwegian Nynorsk

```bash
git clone https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk.git \
  data/raw/UD_Norwegian-Nynorsk
git -C data/raw/UD_Norwegian-Nynorsk checkout \
  aaeb9d90c748c2bd9e272f180b599484f9f05ac6
```

Both datasets are distributed under CC BY-SA 4.0 and retain their own
attribution and share-alike requirements.

## Bokmål training and evaluation

Reproduce the unweighted gold-only student:

```bash
python -m prism.languages.norwegian.train_baseline
```

Reproduce the selected class-weighted ablation without overwriting the
unweighted checkpoint:

```bash
python -m prism.languages.norwegian.train_baseline \
  --checkpoint runs/nb-student-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Evaluate a fixed checkpoint on the development split:

```bash
python -m prism.languages.norwegian.evaluate_baseline \
  --checkpoint runs/nb-student-weighted/best.pt \
  --analysis runs/nb-student-weighted/development-analysis.json
```

The commands intentionally use training and development data only. The test
split is reserved for final evaluation after the model and decision policy are
frozen.

## Tests

Run the complete Python suite from the repository root:

```bash
python -m pytest python/tests
```

Run Ruff and the whitespace check before handing off changes:

```bash
python -m ruff check python
python -m ruff format --check python
git diff --check
```

## Repository layout

```text
Prism/
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/prism/
│   ├── src/
│   ├── tests/
│   ├── umbrella/
│   └── vendor/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   ├── benchmarks.md
│   └── model-strategy.md
├── java/
│   ├── pom.xml
│   └── src/main/java/io/github/dmlux/prism/
├── logos/
├── python/
│   ├── pyproject.toml
│   ├── src/prism/
│   │   ├── data/
│   │   ├── evaluation/
│   │   ├── export/
│   │   ├── languages/
│   │   ├── modeling/
│   │   ├── schema/
│   │   └── training/
│   └── tests/
├── swift/
│   ├── Package.swift
│   ├── Sources/PrismKit/
│   └── Tests/PrismKitTests/
├── Prism.xcworkspace/
└── README.md
```

Datasets, checkpoints, virtual environments, caches, and generated artifacts
are excluded through `.gitignore`.

## Roadmap

1. Calibrate confidence and freeze the language artifact schema.
2. Evaluate the frozen model once on the untouched test splits.
3. Export the selected student and measure the 6,000-token document fixture on
   production runtimes.
4. Provide stable Swift, Java/Kotlin, and C++ packages over the versioned model
   artifact contract.

## Licensing

Prism source code is licensed under the
[Apache License 2.0](LICENSE.md). External datasets and pretrained models
retain their own licenses. Every released model artifact must document its
training-data provenance, dataset license, backbone license, resolved
configuration, and measured metrics separately.
