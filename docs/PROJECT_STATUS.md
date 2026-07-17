# Prism project status

Last updated: 2026-07-17

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

Norwegian Bokmål is the only current implementation target, but the
architecture is explicitly designed for many languages. A
language-independent core owns batching, alignment, reusable task-head
families, training, evaluation, distillation, export, and unified native API
contracts. Replaceable language profiles select the teacher and student
backbones, tokenizer behavior, normalization, annotation and label schemas,
decoding, provenance, and licenses. NorBERT4 is the first Norwegian
configuration, not a dependency of the generic pipeline.

The accepted next-generation direction is a teacher-student architecture: a
high-capacity language-specific teacher is used during training, while only a
compact student is shipped for fast local inference. The first profile uses
Norwegian models and data. The full decision and release gates are in
`docs/model-strategy.md`. The detailed data flow from external tokens through
the Transformer, task heads, distillation, export, and LexKeep inference is
documented in `docs/ARCHITECTURE.md`.

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
- Current export-compatible development PyTorch version: 2.12.1
- Current ExecuTorch version: 1.3.1
- Primary local accelerator: Apple MPS, with CPU fallback
- Source license: Apache License 2.0

The editable environment is installed with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev,export]'
```

The complete test suite is run from the repository root:

```bash
python -m pytest python/tests
```

The complete suite currently contains 85 passing tests, including token
spacing, subword alignment, language profiles, tokenizer batches, backbone
loading, contextual outputs, realignment, task heads, student composition,
target batching, masked multi-task loss, gradient masking, and export-adapter
coverage. Ruff 0.15 provides repository-wide formatting and linting. Python
compatibility is explicitly restricted to Python 3.12.

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

CoNLL-U loading currently retains token text, lemma, UPOS, morphology as a
dictionary of feature names to values, and the `SpaceAfter=No` whitespace
signal needed to reproduce original token boundaries. Dependency information
is present in the source data but is not currently used by the models.

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

The next-generation data and output contract now additionally provides:

- an immutable, versioned morphology schema with deterministic feature and
  atomic-value ordering;
- explicit `<NONE>` labels and deliberate multi-value support;
- validated runtime-independent morphology encoding and decoding;
- deterministic lemma edit rules derived around the longest shared token and
  lemma substring;
- Norwegian Bokmål UD lemma normalization that removes the treebank-specific
  `$` marker without changing the generic lemma-rule contract;
- a versioned UPOS schema and bundled `TokenTaskSchema`;
- a typed `TokenizedBatch` contract;
- a generic pinned `PretrainedBackboneSpec` with NorBERT4-xsmall as the first
  Norwegian configuration;
- a generic `LanguageProfileSpec` that connects a BCP 47 language tag and
  display name to a replaceable student backbone;
- a `prism.languages.norwegian` package that exclusively owns the concrete
  NorBERT4-xsmall configuration and the first Bokmål profile;
- a Fast-Tokenizer loader, byte-level whitespace preparation, first-subword
  alignment, padded token indices, and token masks;
- a language-independent adapter from `PretokenizedSentence` batches to the
  typed `TokenizedBatch`, verified with both unit tests and the real pinned
  NorBERT4 tokenizer;
- a generic pretrained-backbone loader and typed contextual subword and token
  outputs;
- a language-independent forward adapter from `TokenizedBatch` to contextual
  subword vectors, followed by first-subword gathering into one contextual
  vector per externally supplied token;
- a profile-controlled repair path for custom backbones whose non-persistent
  buffers are not restored correctly by Transformers' low-memory loading. The
  pinned NorBERT4-xsmall profile enables this path because its rotary position
  buffers otherwise contain uninitialized values and produce NaNs;
- fail-fast validation that rejects non-finite contextual subword vectors at
  the first Prism output boundary;
- a schema-configured `TokenTaskHeads` module with shared UPOS, per-feature
  morphology, and lemma-rule head implementations whose dimensions come from
  the selected language schema;
- a language-independent `TokenTagger` composition that connects any
  compatible PyTorch backbone to first-subword alignment and the shared task
  heads;
- typed, padded `TokenTaskTargetBatch` construction for variable-length
  supervised sentences, including separate token and usable-lemma masks;
- a differentiable joint loss using masked cross-entropy for UPOS and lemma
  rules plus masked binary cross-entropy for morphology. Tests prove that all
  three tasks receive gradients while padding positions receive zero gradient.

The real pinned NorBERT4-xsmall model has completed the full local path from a
`PretokenizedSentence` through tokenization, backbone inference, and
subword-to-token alignment. The confirmed shapes for the four-token sentence
`Jeg så filmen.` are `[1, 5, 192]` at the subword boundary and `[1, 4, 192]` at
the token boundary, with finite values throughout.

The real pinned NorBERT4-xsmall student has also completed one supervised
AdamW optimization step on the first sentence of the pinned Bokmål training
split. The schema contained 18 morphology features and 622 lemma rules; the
batch contained five tokens. The initial joint loss was `30.290184020996094`.
The summed gradient norms were `2084.5696479082108` for the backbone and
`209.72279049828649` for the task heads, proving that the joint loss propagates
through both parts of the actual student. This is a smoke test, not a quality
benchmark.

The same real five-token optimization step also completed on the local Apple
MPS backend. NorBERT4, token alignment, all schema-sized heads, the joint loss,
backpropagation, and AdamW ran on `mps` with a finite initial loss of
`29.2430362701416`. The difference from the CPU smoke-test loss is expected
with dropout and backend-specific floating-point execution; neither value is a
quality benchmark. The isolated Codex process could not access MPS, so this
result was verified in the repository's local Python 3.12 environment.

A two-batch MPS mini-epoch then completed on the first 16 pinned training
sentences: 258 tokens and 258 usable lemma targets. The token-weighted losses
were `11.401944160461426` for UPOS, `1.705126166343689` for morphology, and
`13.548186302185059` for lemma rules, for a joint loss of
`26.655256628990173`. This proves the integrated shuffling, lazy tokenization,
device transfer, differential AdamW parameter groups, gradient clipping,
linear schedule, and epoch aggregation path. It remains a smoke test, not a
quality benchmark. The unexpectedly large initial task losses must be
investigated before committing to a long training run, including checking the
scale of NorBERT4 token representations and whether the shared heads need an
explicit normalization boundary.

That initial-logit investigation is complete. Fresh NorBERT4 token vectors had
a standard deviation of `6.82661771774292`, mean token L2 norm of
`93.08756256103516`, and maximum absolute value of `23.687984466552734`.
Directly applying default PyTorch linear heads produced UPOS and lemma-logit
standard deviations of `4.661464691162109` and `3.9709372520446777`, with an
initial joint loss of `27.334781646728516`. NorBERT4's own token classifier
also applies a non-affine LayerNorm before its projection. Prism now defines a
shared, language-independent, non-affine LayerNorm boundary before all task
heads. After that change, UPOS and lemma-logit standard deviations fell to
`0.681553065776825` and `0.5837973952293396`; their initial losses fell to
`3.326749324798584` and `6.77863073348999`, and the joint initial loss fell to
`10.851276397705078`. These values are close to the expected random-classifier
scales for 17 UPOS labels and 622 lemma rules. A regression test proves that
all heads are stable under positive affine rescaling of backbone vectors.

A training/development-only exploratory measurement produced 622 normalized
lemma edit rules from 243,885 training tokens. These rules cover 36,336 of
36,369 development-token gold rules, or 99.9093%. This is oracle vocabulary
coverage, not model accuracy. The test split was not used.

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
- Keep the Prism core language-independent. Generic batching, alignment,
  task-head construction, losses, distillation, calibration, evaluation,
  export, and native APIs depend on typed contracts rather than NorBERT4 or a
  concrete language.
- Put backbones, tokenizers, normalization, dataset adapters, schemas, label
  inventories, decoding, provenance, and licenses behind replaceable language
  profiles. The head families remain shared while their dimensions are derived
  from each language schema.
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
- Export versioned model packages around a manifest, vocabularies, labels,
  provenance, licenses, and one or more backend-specific model artifacts.
  ExecuTorch `.pte` is the first portable path; direct Core ML remains an
  explicit Apple comparison. Native libraries do not expose runtime-specific
  types in their public APIs.
- Treat document inference as a release property. The initial gate uses a
  6,000-token, 200-sentence fixture and requires at most 1.0 second median warm
  latency, 1.5 seconds p95 warm latency, 3.0 seconds cold load plus inference,
  250 MiB additional peak memory, and a 100 MiB quantized Norwegian package on
  a recorded Apple Silicon reference machine.

## Immediate next milestone

The data, Transformer input, contextual-output, and token-alignment contracts
are now in place. NorBERT4-xsmall is loaded only through the Norwegian profile,
and a real finite forward pass has been proven.

The early backend-neutral ExecuTorch feasibility spike is complete. The real
NorBERT4-xsmall backbone plus the tensor-only export adapter passed strict
`torch.export`, lowered to a portable ExecuTorch program, and produced an
85,023,300-byte artifact (approximately 81 MiB) outside the repository. The
ExecuTorch 1.3.1 Python runtime executed the exported `forward` method with an
output shape of `[1, 5, 192]`; all values were finite and the maximum absolute
difference from eager PyTorch was `6.4849853515625e-05`.

The prebuilt ExecuTorch 1.3.1 runtime is binary-compatible with PyTorch 2.12.x,
not the previously installed PyTorch 2.13.0, despite its open-ended package
lower bound. Prism therefore pins PyTorch to `>=2.12,<2.13`. Portable runtime
execution is proven; Core ML, Metal, XNNPACK delegation, dynamic shapes,
quantization, and full-student export remain separate measured gates.

The shared task-head and initial trainable-student composition is now complete:
the generic `TokenTagger` connects a replaceable backbone, token alignment, and
schema-sized UPOS, morphology, and lemma-rule heads. Supervised sentences can
be padded into typed target tensors, and the joint masked loss propagates
gradients through every task without learning from padding. The paired
supervised collator and minimal optimizer step are implemented, and the full
Bokmål student has trained for a real two-batch MPS mini-epoch. Device-aware
batching, reproducible shuffling, differential AdamW policies, gradient
clipping, warmup/decay scheduling, and token-weighted epoch loss aggregation
are implemented. The initial-logit scale is resolved through the shared
normalization boundary. The next model milestone is development evaluation and
versioned checkpoint metadata before a long training run.

The generic input, batching, and model code must remain usable by a future
language profile with a different tokenizer and backbone.

### Deferred trigger: Nynorsk

Once the first Bokmål student trains and evaluates end to end reproducibly,
the next scope expansion is Nynorsk (`nn`). This happens before expensive
teacher fine-tuning and final student selection, not before the initial Bokmål
pipeline works.

The Nynorsk work will pin the official UD Norwegian Nynorsk treebank and keep
separate `nb` and `nn` profiles, schemas, lemma rules, calibration, and metrics.
Both profiles may share the NorBERT4 tokenizer and backbone. Development
experiments will compare separate students, a jointly trained encoder with
shared configurable heads, and a jointly trained encoder with
written-standard-specific heads or an equivalent explicit standard signal.
No architecture is selected from a combined Norwegian score alone.

## Longer-term roadmap

Once the output contract is stable:

1. benchmark and select a Norwegian teacher on the development split;
2. train a compact student with and without distillation;
3. add confidence calibration and abstention thresholds;
4. export the fixed student and prove PyTorch-to-ExecuTorch parity;
5. benchmark document inference and integrate the artifact through a Swift
   runtime wrapper for LexKeep;
6. decide on dependency parsing separately after the token tasks are stable;
7. add Nynorsk after the first reproducible Bokmål student and compare shared
   versus written-standard-specific Norwegian model variants;
8. add further languages as separate profiles after Norwegian is stable,
   without copying the core pipeline or changing the unified public API.
