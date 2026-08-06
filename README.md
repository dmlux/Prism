<p align="center">
  <img src="logos/logo.svg" alt="Prism" width="420">
</p>

<p align="center">
  <a href="https://github.com/dmlux/Prism/releases?q=v0&expanded=true"><img src="https://img.shields.io/github/v/release/dmlux/Prism?filter=v*&label=library&color=blue" alt="latest library release"></a>
  <a href="https://central.sonatype.com/artifact/io.github.dmlux/prism"><img src="https://img.shields.io/maven-central/v/io.github.dmlux/prism?label=maven%20central&color=blue" alt="Maven Central"></a>
  <a href="https://github.com/dmlux/Prism/releases?q=prism-no&expanded=true"><img src="https://img.shields.io/github/v/release/dmlux/Prism?filter=prism-no-*&label=model%3A%20norwegian&color=purple" alt="latest Norwegian model release"></a>
  <a href="https://huggingface.co/dmlux/prism-no"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20hub-dmlux%2Fprism--no-yellow" alt="Hugging Face model"></a>
</p>

# Prism

Prism is an open-source NLP toolkit for fast, local, privacy-friendly
linguistic analysis on end-user devices. It tags text with
part-of-speech (UPOS), morphological features, and lemmata — each
decision with a **calibrated confidence** — fully offline, optimized
for small bundles and native performance. The current model covers
Norwegian: Bokmål (`nb`) and Nynorsk (`nn`) in one set of weights.

**Highlights**

- **Built for devices, not servers:** the deployable model is a 45 MB
  artifact tagging ~3,200 tokens/s on a laptop CPU — no GPU, no
  network, no Python.
- **Native APIs for every major platform:** Swift, C++, C, and Java
  (Kotlin-compatible) over one shared model contract, plus the Python
  reference runtime. Three integration routes: PrismKit (Swift),
  PrismNative (Apple projects with a C/C++ core), and CMake/C++ for
  general native platforms.
- **Calibrated confidences:** every tag comes with a probability an
  application can actually act on (fitted temperatures, UPOS ECE
  0.0017).
- **Source mapping:** every token and sentence of a raw-text analysis
  carries half-open UTF-8 byte ranges into the exact input string, so
  applications can highlight results in their own documents — robust
  against the internal text repairs (see
  [docs/INTEGRATION.md](docs/INTEGRATION.md#source-mapping)).
- **Quality that competes:** beats UDPipe 2.17 on UPOS and lemmata on
  the official UD test splits at a twentieth of its size.

## How it works

```mermaid
flowchart TB
    Text["raw text"] --> Seg["sentence segmentation"]
    Seg --> BPE["byte-level BPE subwords"]
    BPE --> Backbone["compact Transformer backbone<br/>(NorBERT4-xsmall, distilled)"]
    Backbone --> Tokens["token representations<br/>+ character-CNN features"]
    Tokens --> Upos["UPOS head"]
    Tokens --> Morph["morphology heads"]
    Tokens --> Lemma["lemma-rule head"]
    Upos --> Out["calibrated probabilities<br/>one result per token"]
    Morph --> Out
    Lemma --> Out
```

The model is a compact Transformer student distilled from a larger
teacher and exported to [ExecuTorch](https://pytorch.org/executorch)
programs with fixed shapes; runtimes pick the smallest program each
batch fits into, so short sentences never pay long-sentence padding.
Decoding policy and calibration are baked into the exported graph —
integrations do argmax and one threshold, nothing more.

Want the full picture? **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
is the complete technical reference behind this diagram — written as a
learning text that follows the data from an incoming token to the
finished prediction: the NorBERT4 block structure, the learned layer
mixture and pooling, the character CNN, the hybrid morphology contract,
the structured decoder, distillation, calibration, and export, with
every number verified against the implementation and diagrams for each
stage.

**Model facts** (prism-no 0.2.3)

| Fact | Value |
| --- | --- |
| Parameters | 17.6 M (16.9 M backbone incl. 9.8 M embedding, 0.7 M heads + character CNN) |
| Architecture | 16-layer encoder-only Transformer (NorBERT4-xsmall: hidden 192, 3 attention heads), distilled from NorBERT4-large |
| Tasks and label spaces | 17 UPOS tags · 18 morphology features · 1,059 lemma edit rules |
| Vocabulary | 51,200 byte-level BPE subwords · 120-character vocabulary for the character CNN |
| Languages | Norwegian Bokmål (`nb`) and Nynorsk (`nn`), one shared set of weights |
| Sentence capacity | up to 96 tokens / 160 subwords per sentence (longer sentences are chunked automatically); fixed batch of 8 |
| Precision variants | fp32 (≈ 94 MB) and int8 "fast" (≈ 45 MB) |
| Training data | gold UD treebanks plus teacher-labeled silver text (NBdigital, municipal documents, Nynorsk Wikipedia) — details below |
| Runtime | ExecuTorch / XNNPACK, CPU only, offline |

## Quality and speed

Evaluated once on the untouched official UD test splits against UDPipe
2.17 (gold tokenization, official CoNLL definitions):

| Test F1 | Prism | UDPipe 2.17 |
| --- | ---: | ---: |
| Bokmål UPOS | **98.76%** | 98.57% |
| Bokmål Lemmas | **98.98%** | 98.87% |
| Bokmål UFeats | 97.20% | **97.59%** |
| Nynorsk UPOS | **98.77%** | 98.60% |
| Nynorsk Lemmas | **98.68%** | 98.56% |
| Nynorsk UFeats | 96.94% | **97.38%** |

**Strengths:** best-in-class UPOS and lemmata; runs locally at
~3,200 tokens/s (C++/Java) on an Apple M4 Max CPU — UDPipe's comparable
models sit behind a ~700 MB server deployment; calibrated confidences;
one model for both written standards, mixed input welcome.

**Weaknesses:** exact morphology bundles (UFeats — every feature of a
word must match) trail UDPipe by ~0.4 pp; no dependency parsing, named
entities, or raw-text sentence splitting beyond the shipped
segmentation policy; one language so far (the architecture and
artifact contract are language-independent by design).

Complete tables — including the fast-versus-fp32 quality gate and the
cross-binding runtime matrix — live in
[docs/benchmarks/prism-no-0.2.2.md](docs/benchmarks/prism-no-0.2.2.md).

## Get a model

Download from [Releases](https://github.com/dmlux/Prism/releases) and
unpack; each model version ships in two precisions, and an application
bundles exactly one:

| Artifact | Size | When to use |
| --- | ---: | --- |
| `prism-no-0.2.3-fast` | ≈ 45 MB | **Recommended.** int8, up to 2× faster, quality within 0.014 pp of fp32 |
| `prism-no-0.2.3` | ≈ 94 MB | Bit-exact fp32 reference behind the published benchmark |

```bash
curl -LO https://github.com/dmlux/Prism/releases/download/prism-no-0.2.3/prism-no-0.2.3-fast.tar.gz
tar -xzf prism-no-0.2.3-fast.tar.gz   # unpacks the prism-no-0.2.3-fast/ folder
```

Unpack it wherever suits your project — the folder's path is what you
hand to the tagger APIs below. Verify downloads against the release's
`SHA256SUMS`. The unpacked directory is everything the runtimes need;
its contents are documented in
[docs/INTEGRATION.md](docs/INTEGRATION.md).

## Quick start

All bindings share the same shape: point a tagger at an artifact
directory, put text in, get sentences of tokens with UPOS, morphology,
lemma, and confidences out. Sensible defaults (thread count, program
selection) are built in. Raw-text results additionally carry
`Utf8ByteRange` source positions — half-open UTF-8 byte ranges into the
exact input string (not UTF-16 offsets; see
[docs/INTEGRATION.md](docs/INTEGRATION.md#source-mapping)).

**The artifact argument is a local filesystem path, not a model ID.**
Unlike Hugging Face-style APIs, nothing is downloaded at runtime:
`"prism-no-0.2.3-fast"` in the snippets below means *the unpacked
folder from the release, addressed relative to your process's working
directory*. In practice you either pass an absolute path, or ship the
folder with your application and resolve it from there — as bundle
resources on macOS/iOS, next to the executable or in your resources
directory on Windows/Linux/JVM. The snippets note the idiomatic place
per platform.

### Swift

Add the Prism package with plain version pinning —
`.package(url: "https://github.com/dmlux/Prism.git", from: "0.5.0")`
(or `exact:` for apps that must never float). Prism embeds the prebuilt
ExecuTorch frameworks as binary targets, so its manifest is fully
version-stable and nothing else needs to be added. A minimal complete
project lives in [examples/swift-quickstart](examples/swift-quickstart);
details in [INTEGRATION.md](docs/INTEGRATION.md).

```swift
import PrismKit

// Ship the artifact folder as a bundle resource (drag it into Xcode as
// a folder reference) and resolve it from the bundle:
let artifactDirectory = Bundle.main.resourceURL!
    .appendingPathComponent("prism-no-0.2.3-fast")

// The default compute device (.automatic) picks the best program the
// artifact ships; current artifacts contain CPU (XNNPACK) programs.
let tagger = try PrismTagger(artifactURL: artifactDirectory)
for sentence in try tagger.tag(text: "Hun kjøpte tre gamle bøker.") {
    for token in sentence.tokens {
        print(token.text, token.upos, token.lemma, token.uposConfidence)
    }
}
```

### Apple apps with a C/C++ core (PrismNative)

For Xcode projects whose core is C, C++, or Objective-C++: add the
Prism package, select the **PrismNative** product, and use the stable
C ABI directly — no CMake, no Python, no manual linker flags or header
paths. The XNNPACK backend and all kernels are baked into the binary
framework.

```cpp
#include <prism/prism_c.h>

prism_tagger* tagger = prism_tagger_create(artifact_directory_path);
prism_result* result = prism_tagger_tag_text(tagger, "Hun kjøpte tre gamle bøker.");
/* … see docs/INTEGRATION.md for the complete quick start … */
prism_result_destroy(result);
prism_tagger_destroy(tagger);
```

Ship the model folder as a bundle resource and pass its absolute path;
details, platform matrix, and the release checksum flow live in
[docs/INTEGRATION.md](docs/INTEGRATION.md).

### C++

Build with CMake (`cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release`),
link the aggregate target: `target_link_libraries(app PRIVATE prism)`.

```cpp
#include <prism>

// The argument is a local directory path. A bare name like this is
// resolved relative to the working directory of your process — for
// anything beyond experiments, build an absolute path (for example
// from your executable's location or your app's data directory).
prism::tagger::Tagger tagger("prism-no-0.2.3-fast");
for (const auto& sentence : tagger.TagText("Hun kjøpte tre gamle bøker.")) {
    for (const auto& token : sentence.tokens) {
        std::cout << token.text << " " << token.upos << " " << token.lemma
                  << " " << token.upos_confidence << "\n";
    }
}
```

### C

For application cores that link plain C (or any foreign-function
interface):

```c
#include <prism/prism_c.h>

/* Local directory path, resolved like any relative path in C —
 * against the process working directory. Prefer an absolute path. */
prism_tagger* tagger = prism_tagger_create("prism-no-0.2.3-fast");
prism_result* result = prism_tagger_tag_text(tagger, "Hun kjøpte tre gamle bøker.");
for (size_t t = 0; t < prism_result_token_count(result, 0); ++t) {
    printf("%s %s %s %.3f\n",
        prism_result_token_text(result, 0, t),
        prism_result_token_upos(result, 0, t),
        prism_result_token_lemma(result, 0, t),
        prism_result_token_upos_confidence(result, 0, t));
}
prism_result_destroy(result);
prism_tagger_destroy(tagger);
```

### Java (and Kotlin)

Prism is on [Maven Central](https://central.sonatype.com/artifact/io.github.dmlux/prism)
with embedded native libraries for macOS (arm64, x86_64) and Linux
(x86_64, aarch64) — the right one is extracted and loaded
automatically, no further setup:

Maven (`pom.xml`):

```xml
<dependency>
  <groupId>io.github.dmlux</groupId>
  <artifactId>prism</artifactId>
  <version>0.5.0</version>
</dependency>
```

Gradle (`build.gradle.kts`, or the same line without parentheses in a
Groovy `build.gradle`):

```kotlin
implementation("io.github.dmlux:prism:0.5.0")
```

The same JAR is attached to each [`v*` release](https://github.com/dmlux/Prism/releases)
for manual use without a build tool. On other platforms, build
`prism.jar` plus `libprism_jni` yourself (one CMake run) and load via
`-Djava.library.path=...` or `PrismTagger.loadNativeLibrary(...)`;
packaging and the JNI mechanics are explained in
[INTEGRATION.md](docs/INTEGRATION.md).

```java
import io.github.dmlux.prism.PrismTagger;

// The Path must point to a real directory on disk. Ship the artifact
// in your application's resources/installation directory — note that
// a folder packed *inside* a JAR is not a filesystem path; extract it
// to a data directory on first run, then load from there.
Path artifact = Path.of("prism-no-0.2.3-fast");

try (PrismTagger tagger = PrismTagger.load(artifact)) {
    for (var sentence : tagger.tagText("Hun kjøpte tre gamle bøker.")) {
        for (var token : sentence.tokens()) {
            System.out.println(token.text() + " " + token.upos() + " "
                + token.lemma() + " " + token.uposConfidence());
        }
    }
}
```

### Python

The Python package is the research and reference runtime: it runs the
frozen *checkpoint* eagerly through PyTorch (a trained-weights file
under `runs/`, produced by training — not the released artifact), and
it is the implementation every native binding is parity-tested against.
Use it for research, evaluation, and training; use the native bindings
for applications.

**Recommended: Python 3.12** — it has the broadest compatibility with
the current PyTorch/ExecuTorch ecosystem. Set up an isolated virtual
environment inside the repository (a "venv" keeps Prism's dependencies
out of your system Python; the activate step is per terminal session):

```bash
git clone https://github.com/dmlux/Prism.git
cd Prism
python3.12 -m venv .venv          # create the environment once
source .venv/bin/activate         # activate it (Windows: .venv\Scripts\activate)
python -m pip install -e './python[dev]'
```

```python
from prism.languages.norwegian.tagger import NorwegianTagger

tagger = NorwegianTagger(
    checkpoint_path=checkpoint,     # e.g. runs/<run>/best-development-task-accuracy.pt
    calibration_path=calibration,   # the calibration.json fitted for it
)
for sentence in tagger.tag_text("Hun kjøpte tre gamle bøker."):
    for token in sentence.tokens:
        print(token.text, token.upos, token.lemma, token.upos_confidence)
```

Training your own checkpoint (or reproducing the released one) is
covered in [docs/TRAINING.md](docs/TRAINING.md).

Every binding also accepts pretokenized input (`tag(pretokenized:)`,
`TagPretokenized`, `tagPretokenized`, `prism_tagger_tag_tokens`) when
your application already has words.

## Bring your own language

Norwegian is the first language, not the last: the model, training,
export, and artifact contracts are language-independent, and every
runtime reads any conforming artifact without code changes. If you have
a [Universal Dependencies](https://universaldependencies.org/)-style
treebank (CoNLL-U with `FORM`, `LEMMA`, `UPOS`, `FEATS`) and a Hugging
Face encoder for your language, you can train and ship your own Prism
model: label schemas, morphology features, and lemma rules are derived
from your data automatically. The complete guide — data format and
placement, every pipeline stage from teacher to released artifact, and
the language-profile mechanism — is
[docs/TRAINING.md](docs/TRAINING.md).

## Training data and licenses

Prism source code is licensed under the
[Apache License 2.0](LICENSE.md). Model weights are released under
**CC BY-SA 4.0**: using or bundling the unmodified artifact — including
commercially, in closed-source applications — is fine (keep the
attribution); redistributed modified weights must stay open.

The Norwegian model exists thanks to openly licensed resources: the
[UD Norwegian-Bokmaal](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal)
and [UD Norwegian-Nynorsk](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk)
treebanks (Universal Dependencies contributors, based on the Norwegian
Dependency Treebank by the National Library of Norway, CC BY-SA 4.0),
Språkbanken's NBdigital and municipal-documents corpora (National
Library of Norway, CC0), the Nynorsk Wikipedia (CC BY-SA 4.0), and the
[`ltg/norbert4-xsmall`](https://huggingface.co/ltg/norbert4-xsmall)
backbone (Language Technology Group, University of Oslo, Apache 2.0).
Pinned revisions and checksums travel inside every artifact
(`manifest.json`, `LICENSES/`).

## Documentation

- [docs/INTEGRATION.md](docs/INTEGRATION.md) — the artifact contract
  and per-binding integration details
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the model and the
  native inference stack are designed, explained from the ground up
- [docs/benchmarks/](docs/benchmarks/) — quality and runtime benchmarks
  per released artifact; [docs/BENCHMARKS.md](docs/BENCHMARKS.md) holds
  the model-development history
- [docs/TRAINING.md](docs/TRAINING.md) — training models and adding new
  languages, from data format to released artifact
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — building and testing
  Prism itself
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — every accepted
  decision and milestone, in order

## Roadmap

1. Close the exact-morphology gap (joint bundle consistency).
2. More languages over the same language-independent contracts.
3. Runtime follow-ups: newer ExecuTorch pin for the C++ build,
   GPU-lowered artifact variants, Hugging Face mirrors of the releases.
