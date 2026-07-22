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
morphology, and UPOS also improve on both written standards. The 69.9 MB
checkpoint remains below the 100 MB target. Decoupled knowledge distillation
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
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   ├── benchmarks.md
│   └── model-strategy.md
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
