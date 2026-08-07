# Training Prism models

The complete path from raw treebank to released artifact, and what it
takes to train a **new language**. Prism is open source — this page is
written so that someone with suitable training data can reproduce the
Norwegian model or build their own.

Environment setup (Python 3.12, virtualenv, dependencies) is described
in [DEVELOPMENT.md](DEVELOPMENT.md); the design rationale behind every
stage in [ARCHITECTURE.md](ARCHITECTURE.md); every accepted decision in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

## The training data format

Prism trains on **CoNLL-U** files following the
[Universal Dependencies](https://universaldependencies.org/) (UD)
conventions — the format of every official UD treebank. Three files per
treebank: a training, a development, and a test split.

Prism reads these columns per token line:

| Column | Used for |
| --- | --- |
| `FORM` | the token text (model input) |
| `LEMMA` | lemma target — converted into a **lemma edit rule** (prefix/suffix removal and addition relative to the form); `_` marks unannotated lemmata and is excluded from the loss |
| `UPOS` | the universal part-of-speech target |
| `FEATS` | the morphology target: `Name=Value` pairs separated by `\|`, multiple values per feature separated by `,` |
| `MISC` | `SpaceAfter=No` reconstructs the original spacing |

Everything else (XPOS, HEAD, DEPREL, DEPS) is ignored — Prism does not
train a parser. Multi-word-token ranges and empty nodes follow UD
semantics. Label spaces (UPOS labels, morphology features and values,
lemma rules, character vocabulary) are **derived from the training
split only** — the schema is data-driven, nothing is hard-coded per
language.

If your data is not UD: any corpus you can convert into these five
columns works. The quality ceiling is set by annotation consistency —
UPOS/FEATS/LEMMA conventions must be applied uniformly, and the
development split must follow the same conventions as training.

## Where data lives

Pinned treebanks are cloned under the git-ignored `data/raw/` directory
and referenced from the language profile with an exact revision:

```bash
git clone https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal.git \
  data/raw/UD_Norwegian-Bokmaal
git -C data/raw/UD_Norwegian-Bokmaal checkout 396d11f0c2bd290a2a2711015c04ac25bc3dcc06

# English (Ettin profile): UD_English-EWT gold treebank
git clone https://github.com/UniversalDependencies/UD_English-EWT.git \
  data/raw/UD_English-EWT
git -C data/raw/UD_English-EWT checkout 4a4d77f599ea53cc405f85d0cec4b2f14f81d42b
```

Prepared silver corpora live under `data/processed/<source>/`;
checkpoints and analyses under `runs/<run-name>/`; exported artifacts
under `models/`. None of these directories are committed.

## The pipeline, stage by stage

All commands run from the repository root inside the virtualenv. The
Norwegian entry points live under `prism.languages.norwegian.*`;
defaults reproduce the selected production configuration, so plain
invocations are the accepted recipe.

### 1. Train the teacher

A larger backbone trained on gold data only; its soft targets later
supervise the student:

```bash
python -m prism.languages.norwegian.train_baseline \
  --model-role teacher \
  --teacher-backbone large \
  --checkpoint runs/no-teacher-large/best.pt
```

### 2. (Optional but selected) Prepare and label silver data

Large unlabeled corpora, labeled once by the frozen teacher, extend the
gold data. Preparation streams the source archive, keeps supplied
sentence boundaries, filters low-confidence OCR and over-long
sentences, deduplicates, and excludes anything overlapping any UD
split:

```bash
python -m prism.languages.norwegian.prepare_silver_corpus \
  --source nbdigital-nob \
  --archive <downloaded-archive> \
  --output data/processed/nbdigital-nob-free \
  --manifest data/processed/nbdigital-nob-free/manifest.json
```

Offline teacher labeling binds hard pseudo-labels plus per-task
confidences to the corpus manifest and teacher checkpoint (an optional
second checkpoint enables the agreement filter):

```bash
python -m prism.languages.norwegian.label_silver_corpus \
  --silver-corpus data/processed/nbdigital-nob-free/pretokenized.jsonl \
  --silver-manifest data/processed/nbdigital-nob-free/manifest.json \
  --checkpoint runs/no-teacher-base/best.pt \
  --calibration <teacher-calibration.json> \
  --output-directory data/processed/nbdigital-nob-free/labels
```

### 3. Train the student with distillation

The compact deployment model. Gold supervision plus teacher soft
targets (and silver batches when provided):

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --teacher-checkpoint runs/no-teacher-base/best.pt \
  --silver-corpus data/processed/nbdigital-nob-free/pretokenized.jsonl \
  --silver-labels data/processed/nbdigital-nob-free/labels \
  --checkpoint runs/no-student/best.pt
```

Architecture and objective knobs (`--task-head-architecture`,
`--token-pooling`, `--backbone-layer-aggregation`,
`--categorical-distillation-objective`, per-task distillation
temperatures and weights, morphology class-weight caps, silver loss
weight and confidence thresholds) all default to the selected
production configuration; every alternative was gated by a measured
ablation recorded in [PROJECT_STATUS.md](PROJECT_STATUS.md).

### 4. Evaluate on the development split

```bash
python -m prism.languages.norwegian.evaluate_baseline \
  --checkpoint runs/no-student/best-development-task-accuracy.pt \
  --analysis runs/no-student/development-analysis.json
```

Exact, per-label, and Rare/OOV metrics. **The test split is evaluated
exactly once**, after every decision is frozen on development — that
one-shot policy is what makes the published numbers trustworthy.

### 5. Fit the confidence calibration

Per-head temperatures on the development split at the production
logit-correction strength; argmax-invariant, so predictions do not
change:

```bash
python -m prism.languages.norwegian.calibrate_baseline \
  --checkpoint runs/no-student/best-development-task-accuracy.pt \
  --calibration runs/no-student/calibration.json
```

### 6. Export the versioned artifact

Lowers the frozen checkpoint to ExecuTorch programs with the decoding
policy and calibration baked in, records parity fixtures, and writes
the manifest with checksums (contract: [INTEGRATION.md](INTEGRATION.md)):

```bash
python -m prism.languages.norwegian.export_artifact \
  --checkpoint runs/no-student/best-development-task-accuracy.pt \
  --artifact-version 0.3.0 \
  --calibration runs/no-student/calibration.json \
  --small-shapes 24 16 --small-shapes 48 32 --small-shapes 96 64 \
  --external-data
```

Add `--precision int8` (and a `-fast` suffix in the version string) for
the quantized variant; its runtime parity is gated by the C++ test
suite because the Python ExecuTorch wheel ships no quantized kernels.

### 7. Verify through the native runtimes

```bash
ctest --test-dir cpp/build --output-on-failure   # C++, C ABI, Java
cd swift && swift test                            # PrismKit
```

The suites execute the exported programs against the recorded fixtures
and the reference decisions.

## Adding a new language

The model, training, evaluation, export, and artifact contracts are
language-independent; the native runtimes read any conforming artifact
without code changes. A new language needs four things:

1. **Data** — a UD-style treebank (format above) with train/dev/test
   splits, pinned to an exact revision. Check the license permits your
   intended distribution; the artifact records provenance and license.
2. **A backbone** — a Hugging Face encoder for your language (or a
   multilingual one), pinned to a revision. Two sizes work best: a
   compact student for deployment and a larger teacher for
   distillation. For Norwegian these are `ltg/norbert4-xsmall` and
   `ltg/norbert4-large` (with `ltg/norbert4-base` as a registered
   alternate teacher; the released model was distilled from large).
   English is the second reference profile
   (`prism/languages/english/`): the Ettin encoder suite
   (`jhu-clsp/ettin-encoder-17m` student, `jhu-clsp/ettin-encoder-400m`
   teacher). A first-class `transformers` architecture such as ModernBERT
   needs no `trust_remote_code`; set the spec's
   `attention_implementation="eager"` and
   `config_overrides=(("reference_compile", False),)` for a portable
   ExecuTorch export graph.
3. **A language profile** — a `LanguageProfileSpec` wiring tags, names,
   backbones, and treebank paths together
   (`python/src/prism/languages/<language>/profile.py`):

   ```python
   SWEDISH_TREEBANK = UniversalDependenciesTreebankSpec(
       repository_id="UniversalDependencies/UD_Swedish-Talbanken",
       revision="<commit>",
       license_id="CC-BY-SA-4.0",
       training_path=Path("data/raw/UD_Swedish-Talbanken/sv_talbanken-ud-train.conllu"),
       development_path=Path(".../sv_talbanken-ud-dev.conllu"),
       test_path=Path(".../sv_talbanken-ud-test.conllu"),
   )

   SWEDISH_PROFILE = LanguageProfileSpec(
       language_tag="sv",
       display_name="Swedish",
       student_backbone=STUDENT_BACKBONE,   # PretrainedBackboneSpec
       teacher_backbone=TEACHER_BACKBONE,
       gold_treebank=SWEDISH_TREEBANK,
   )
   ```

4. **The language package** — today the pipeline entry points
   (`train_baseline`, `evaluate_baseline`, `calibrate_baseline`,
   `export_artifact`, the silver tooling) live under
   `prism/languages/norwegian/` and delegate almost everything to the
   shared `prism.data`/`prism.modeling`/`prism.training`/
   `prism.exporting` machinery. A new language starts as a copy of that
   package with three genuinely language-specific pieces:
   - **lemma normalization** — Norwegian normalizes UD lemma
     conventions before deriving edit rules
     (`prism/data/norwegian.py`); your language may need its own rules
     or none;
   - **the runtime segmentation policy** — chiefly the abbreviation
     list that protects sentence boundaries (`f.eks.`, `bl.a.` …);
   - **anything your treebank annotates unusually** (Norwegian's
     written-standard split into `nb`/`nn` profiles is an example of
     profile-level modeling).

   Factoring these entry points into fully generic, profile-driven
   commands is on the roadmap; until then the Norwegian package is the
   reference implementation to copy.

Then run the pipeline above with your language's entry points. The
schema (labels, features, lemma rules, character vocabulary) derives
from your training data automatically; the exported artifact works in
every Prism runtime as-is. For a first quality assessment you do not
need a teacher or silver data — a gold-only student (skip stages 1–2,
drop the distillation flags) already produces a complete artifact.

**What to expect:** the Norwegian student trains in hours on an Apple
M-series machine (MPS); silver labeling is the most expensive optional
stage. Quality lives and dies with annotation consistency and
development-split discipline — keep the test split untouched until
everything is frozen, and evaluate it exactly once.
