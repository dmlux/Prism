# Prism project status

Last updated: 2026-07-15

This document is the durable handoff for continuing Prism in a new Codex task.
Read it together with `README.md`, `docs/benchmarks.md`, and the root
`AGENTS.md`. Verify paths and Git state before relying on details that can
change over time.

## Goal

Prism is intended to become a modular, open-source NLP toolkit for fast,
local, and privacy-friendly linguistic analysis. The long-term feature set is:

- tokenization and sentence segmentation;
- part-of-speech tagging;
- lemmatization;
- morphological analysis;
- language-specific models behind a unified API;
- native Swift packages suitable for applications such as LexKeep.

Only Norwegian Bokmål is currently in scope. The accepted next-generation
direction is a teacher-student architecture: a high-capacity pretrained
Norwegian teacher is used during training, while only a compact student is
shipped for fast local inference. The full decision and release gates are in
`docs/model-strategy.md`.

## Collaboration requirements

The user is learning ML while building the project. Continue in small,
understandable increments:

1. Give exactly one concrete next step.
2. Briefly explain what it accomplishes and why it is needed.
3. Wait for the user's result before continuing.

When the user directly asks Codex to edit, inspect, fix, or verify the
repository, Codex should complete that bounded task itself. New ML concepts,
tensor shapes, loss functions, padding behavior, and architectural connections
should be explained in beginner-friendly German.

## Repository and environment

- Repository path after the rename: `/Users/dmlux/git/Prism`
- Project name: Prism
- Python distribution: `prism-nlp`
- Python import package: `prism`
- Supported development runtime: Python 3.12
- Confirmed local Python version used for benchmarks: 3.12.13
- PyTorch version used for benchmarks: 2.13.0
- Primary local accelerator: Apple MPS, with CPU fallback
- Source license: Apache License 2.0

The editable environment is installed with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev]'
```

The complete test suite is run from the repository root:

```bash
python -m pytest python/tests
```

At the time of this handoff, the suite contains nine passing tests.

## Dataset

The current experiments use the official Universal Dependencies Norwegian
Bokmål treebank:

- repository: `UniversalDependencies/UD_Norwegian-Bokmaal`;
- pinned commit: `396d11f0c2bd290a2a2711015c04ac25bc3dcc06`;
- license: CC BY-SA 4.0;
- training split: 15,696 sentences and 243,886 tokens;
- development split: 2,409 sentences;
- test split: 1,939 sentences.

The ignored local dataset path is:

```text
data/raw/UD_Norwegian-Bokmaal/
```

CoNLL-U loading currently retains token text, lemma, UPOS, and morphology as a
dictionary of feature names to values. Dependency information is present in
the source data but is not currently used by the models.

The dataset is not committed. Any separately distributed trained model must
carry dataset provenance, attribution, share-alike review, configuration, and
its own clearly stated model license.

## Completed implementation

Shared data reading remains directly under `python/src/prism/`. The completed
recurrent experiments are isolated under
`python/src/prism/baselines/recurrent/`, including their CLI implementations.
Together, the current Python code provides:

- a CoNLL-U reader and typed token representation;
- deterministic word, POS-tag, character, and morphology vocabularies;
- a most-frequent-tag dictionary baseline;
- sentence encoding into word and POS IDs;
- character encoding with unknown-character handling;
- dynamic batch-local padding for tokens and characters;
- packed variable-length sentence and word processing;
- a word-only BiLSTM POS tagger;
- a word-and-character BiLSTM POS tagger;
- a shared word-and-character multi-task encoder with separate POS and
  morphology output heads;
- training, checkpointing, evaluation, and prediction entry points;
- precision, recall, F1, support, vocabulary-status, and annotated-feature
  evaluation.

The most-frequent-tag comparison is isolated separately in
`python/src/prism/baselines/dictionary.py`. Internal baseline code and tests
import the baseline packages directly. No duplicate top-level model or command
wrappers remain. This gives the next-generation model a clean package namespace
while recurrent checkpoints remain loadable through the explicit baseline
commands.

Prediction currently expects tokens to be supplied externally. This matches
LexKeep's current ownership of tokenization and offsets. A later high-level
Prism API may add language-aware tokenization and sentence segmentation, but
that work is separate from token-level POS and morphology prediction.

## Architecture

The strongest current model path is:

```text
word ID -> word embedding --------------------------+
                                                     +-> sentence BiLSTM
characters -> embeddings -> character BiLSTM -------+         |
                                                               +-> POS head
                                                               +-> Number head
```

Important details:

- Words occurring fewer than twice in training map to `<UNK>`.
- The character encoder still sees their spelling and substantially improves
  unknown-word tagging.
- Padding is computed only to the longest sentence and word in each batch. It
  does not depend on knowing a global maximum sentence length in production.
- Packed sequences prevent the sentence BiLSTM from treating padded positions
  as real context.
- The current morphology head predicts only the UD feature `Number`.
- Missing Number annotations are represented by `<NONE>`.
- Combined feature values such as `Plur,Sing` are currently kept as a single
  rare class and are not learned reliably.

Current default dimensions are:

- word embedding: 64;
- character embedding: 32;
- character BiLSTM hidden size: 32 per direction;
- sentence BiLSTM hidden size: 128 per direction;
- UPOS classes: 17;
- word vocabulary: 12,390 entries at minimum frequency 2;
- character vocabulary: 115 entries.

## Confirmed benchmarks

All numbers use official gold tokenization and the original UD splits. Exact
details live in `docs/benchmarks.md`.

| Model | Development POS | Test POS | Additional result |
| --- | ---: | ---: | --- |
| Most-frequent word dictionary baseline | 90.28% | - | 94.27% known and 53.24% unknown on development |
| Word-only BiLSTM | 91.91% | 91.22% | 66.85% on test `<UNK>` tokens |
| Word + character BiLSTM | 96.76% | 96.00% | 88.87% on test `<UNK>` tokens |
| POS + Number multi-task BiLSTM | 96.49% | 95.96% | 97.27% Number overall on test |

For the selected POS + Number checkpoint:

- checkpoint selected after epoch 6 by the lowest combined development loss;
- development Number accuracy: 97.50% overall;
- development Number accuracy on annotated tokens: 95.78%;
- test Number accuracy: 97.27% overall;
- test Number accuracy on annotated tokens: 95.69%.

The `<NONE>` class is frequent, so overall Number accuracy must never be
reported without the annotated-token score. Rare values also require class-
level metrics; for example, the two test instances of `Plur,Sing` currently
have zero F1.

Training the character POS model for ten epochs showed diminishing returns and
overfitting after the best development region. More epochs alone are not a
reliable path to higher quality. Checkpoint selection must continue to use the
development split, not the test split.

## Checkpoints

Expected ignored local checkpoints are stored below
`models/norwegian-bokmaal/`, including:

```text
pos_bilstm.pt
pos_character_bilstm_10_epochs.pt
pos_number_bilstm.pt
```

Checkpoints contain the model state, word, character, tag, and feature
vocabularies as applicable, architecture dimensions, selected epoch, and
development metrics. They do not contain executable model class definitions;
loading reconstructs the appropriate class from Prism source code and applies
the saved state dictionary.

The rename from Vexo to Prism changed imports and CLI module names but did not
invalidate the existing state-dictionary checkpoints. The POS + Number
checkpoint was successfully evaluated after the rename.

## Important design decisions

- Keep externally supplied token support because LexKeep already tokenizes and
  tracks original offsets.
- Treat tokenization, sentence segmentation, phrase recognition, named-entity
  recognition, and multiword-expression recognition as separate future tasks.
- Use one shared contextual encoder with separate task heads for related token
  predictions. Morphology is not a second model run before or after POS; both
  heads read the same contextual token representation during joint training.
- Preserve development and test separation. Once the test set has informed a
  choice, record the result but do not tune repeatedly against it.
- Favor trustworthy learning-product output: add calibrated confidence and an
  explicit uncertain state before presenting predictions as authoritative.
- Plan for modular Swift packages such as `PrismKit`, `PrismCore`, and
  `PrismNorwegian`, without forcing Swift concerns into the training code
  prematurely.
- Keep datasets and model artifacts outside the source repository. Publish
  models separately with reproducibility and license documentation.
- Use the current recurrent models as measured baselines, not as the final
  production architecture. Do not remove their training or checkpoint support
  until a replacement is reproducible and passes the new quality and runtime
  gates.
- The first next-generation production bundle jointly predicts UPOS, supported
  Norwegian UD morphology features, lemmas, and calibrated confidence. It does
  not silently add dependency parsing, tokenization, sentence segmentation,
  named entities, phrases, or multiword expressions.
- Train a high-capacity Norwegian teacher and distill its predictions into a
  compact student. Only the student is distributed to applications. Compare
  the distilled student against the same architecture trained without
  distillation.
- Export versioned model packages initially around an ExecuTorch `.pte`
  artifact plus a manifest, vocabularies, labels, provenance, and licenses.
  Native libraries wrap this artifact without exposing runtime-specific types
  in their public APIs.
- Treat document inference as a release property. The initial gate uses a
  6,000-token, 200-sentence fixture and requires at most 1.0 second median warm
  latency, 1.5 seconds p95 warm latency, 3.0 seconds cold load plus inference,
  250 MiB additional peak memory, and a 100 MiB quantized Norwegian package on
  a recorded Apple Silicon reference machine.

## Immediate next milestone

The next milestone is the data and output contract for the first
next-generation production bundle. Before selecting or fine-tuning a teacher,
define and test the complete Norwegian morphology representation and lemma
edit-rule representation consumed by both teacher and student.

Morphology uses one head per feature with an explicit `<NONE>` value. The
representation must handle genuinely multi-valued annotations deliberately
instead of encoding a rare comma-joined class by accident. Lemmatization begins
with learned edit rules and must report coverage and unknown-token accuracy.
The resulting schema is versioned and becomes part of checkpoint and exported
artifact metadata.

## Longer-term roadmap

Once the output contract is stable:

1. benchmark and select a Norwegian teacher on the development split;
2. train a compact student with and without distillation;
3. add confidence calibration and abstention thresholds;
4. export the fixed student and prove PyTorch-to-ExecuTorch parity;
5. benchmark document inference and integrate the artifact through a Swift
   runtime wrapper for LexKeep;
6. decide on dependency parsing separately after the token tasks are stable;
7. add further languages as separate modules after Norwegian is stable.
