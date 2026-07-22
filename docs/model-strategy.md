# Prism model and runtime strategy

Status: active architectural direction with selected shared Norwegian student
Last updated: 2026-07-21

## Purpose

Prism should provide serious, locally runnable language models rather than a
collection of isolated training demonstrations. The first production target
remains Norwegian Bokmål, externally tokenized by a caller such as LexKeep.
The architecture must later allow separate language packages behind the same
public API.

This document defines the active Transformer, teacher-student, and native
runtime direction. The gold-only Bokmål student is implemented and measured;
every further quality claim still requires a recorded benchmark on pinned
data splits.

## Language-independent core and language profiles

Norwegian Bokmål is the first implementation, not a structural assumption of
Prism. The architecture is split into a language-independent core and
replaceable language profiles.

The core owns stable mechanisms:

- typed token, batch, prediction, confidence, and artifact contracts;
- subword-to-token alignment and document batching;
- reusable UPOS, per-feature morphology, lemma, and confidence head families;
- loss calculation, distillation, calibration, evaluation, export, and native
  runtime integration;
- the unified Python, Swift, Java/Kotlin, and C++ API semantics.

A language profile supplies replaceable language decisions:

- language and locale identifiers;
- teacher and student backbone specifications;
- tokenizer, whitespace, and normalization behavior;
- dataset adapters and supported annotation schemas;
- UPOS, morphology, and lemma-rule label inventories;
- language-specific decoding, provenance, licenses, and benchmark identity.

The head implementations are shared, but their output sizes are configured by
the selected language schema. For example, every language can use the same
per-feature morphology classifier implementation while exposing a different
set of features and values. Generic task heads must therefore never hard-code
the 18 Norwegian features, 17 UPOS labels, the current shared Norwegian
lemma-rule count, or a NorBERT hidden size.

Generic batching, alignment, model composition, training, evaluation, and
export code depend only on typed backbone and language-profile contracts. A
Norwegian package may select NorBERT4; another language may select a different
tokenizer, teacher, and student without forking the Prism pipeline. NorBERT4 is
the first configuration of this boundary, not the boundary itself.

Teacher and student architectures may differ within one language profile, and
different languages may choose entirely different backbone families. Every
released student still conforms to the same versioned Prism tensor and
artifact contract so native clients do not need model-specific APIs.

## Norwegian written-standard expansion

Prism's Norwegian scope is intended to cover both Bokmål (`nb`) and Nynorsk
(`nn`). NorBERT4 can be shared as a backbone candidate because its official
model card states that it was pretrained on Bokmål, Nynorsk, and Northern Sámi.
Shared pretraining is not evidence of equal downstream quality, so every
written standard requires its own evaluation.

The first end-to-end training path remains Bokmål so the new pipeline can be
debugged without changing multiple datasets at once. As soon as the first
Bokmål student can be trained and evaluated reproducibly, and before expensive
teacher fine-tuning or final student selection, Prism will add the official UD
Norwegian Nynorsk treebank as a separately pinned dataset.

Bokmål and Nynorsk are separate language profiles even when they reference the
same tokenizer and backbone. Each profile owns its own:

- BCP 47 language tag;
- pinned datasets and split identities;
- observed morphology and lemma-rule schemas;
- decoding and confidence calibration;
- per-task quality and document-performance reports.

These profiles are data and evaluation boundaries, not a requirement to ship
two sets of weights. The production target is one shared Norwegian teacher and
one shared compact student that can process Bokmål, Nynorsk, and mixed written
input without a required document-level standard selector. Separately trained
students remain necessary reference experiments for detecting whether joint
training harms either written standard.

The reusable task-head implementations remain shared. The Norwegian
experiments must compare at least:

1. separate Bokmål and Nynorsk students;
2. one jointly trained encoder with shared configurable heads;
3. one jointly trained encoder with small written-standard-specific heads or
   an equivalent explicit written-standard signal.

Selection is based on separate Bokmål and Nynorsk development metrics, model
size, and document inference performance. A combined Norwegian score must not
hide a regression in either written standard. The first Nynorsk target is the
standard written-language treebank; the spoken/dialectal NynorskLIA corpus is a
separate later decision.

Primary references:

- [NorBERT4 model card](https://huggingface.co/ltg/norbert4-base)
- [UD Norwegian Nynorsk](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk)

## Current student and remaining production gap

The implemented NorBERT4-xsmall student already predicts UPOS, the observed
Norwegian UD morphology inventory, and lemma edit rules through shared task
heads. Its gold-only, class-weighted checkpoint is the reproducible control for
later distillation. Threshold-independent development evaluation confirms that
class weighting improves label ranking rather than merely increasing output
volume.

The selected format-3 student uses Mean pooling over every original token's
contiguous contextualized subword span. A controlled comparison against First
pooling held the backbone, schema, heads, losses, optimizer, seed, and data
constant. Mean pooling reduced development loss and improved lemma accuracy
and morphology micro F1 on both Bokmål and Nynorsk, so it is the default for
new training runs. Checkpoints record the policy; older format-3 checkpoints
without that field remain explicitly First pooling for compatibility.

The selected student uses a learned scalar mixture of the final four backbone
layers before Mean pooling. Four softmax-normalized mixture logits and one
learned scale add only five parameters. This improves Loss, UPOS, Lemma,
morphology precision, micro F1, and macro F1 on both Norwegian written
standards. Checkpoints record the policy; older checkpoints without the field
remain explicitly final-layer-only. A strict `torch.export` capture, XNNPACK
lowering to ExecuTorch `.pte`, and runtime parity smoke test cover the complete
fixed-shape tagger with alignment and all task logits. They confirm that the
layer mixture does not introduce a new export blocker. Dynamic shapes,
production backend parity, peak memory, and document-scale performance remain
release requirements.

The selected student adds one shared residual
`Linear(H -> 2H) -> GELU -> Dropout -> Linear(2H -> H)` projection before the
schema-driven linear task heads. The controlled
`linear` versus `shared-mlp` ablation improved every reported headline metric
on Bokmål and Nynorsk, and the subsequent width ablation selects the
`wide-shared-mlp` projection. The later structured-morphology ablation selects
The selected default for new Norwegian training runs is now
`wide-shared-mlp-structured-morphology-character-cnn`. Checkpoints record the
architecture, and old format-3 checkpoints default to `linear`.
Eight epochs improve every headline
metric over the five-epoch control on both written standards without changing
model size or inference cost. Ten epochs improve every headline metric again.
The final predeclared twelve-epoch run then improves Loss, Lemma, and all
morphology summaries on both standards, with only a 0.0128-point Nynorsk UPOS
tradeoff. Twelve epochs are selected as the default, and epoch-count tuning on
these Development splits is now closed before further architecture work and
format-3 teacher training.

The selected shared-projection capacity is `wide-shared-mlp`. It preserves the
residual path and expands the shared token projection from `H` to `2H` before
projecting back to `H`. For the xsmall
Norwegian student this is `192 -> 384 -> 192`; the block contains 148,032
parameters while leaving the schema-driven output heads unchanged. Compared
with the selected 37,056-parameter shared projection, the increase is 110,976
parameters, or
approximately 444 KB in FP32. The existing `shared-mlp` implementation and
checkpoint interpretation remain unchanged. The wider variant improves lemma
accuracy, morphology micro F1, and morphology Average Precision on both
written standards. Bokmål Loss also improves substantially; Nynorsk Loss
worsens despite improved discrete predictions and ranking quality, so raw
confidence calibration remains an explicit release gap rather than being
hidden by the selection.

It is not yet a production release because the selected format-3-distilled
Student still requires confidence calibration and frozen artifact metadata;
native runtime packaging and the 6,000-token document benchmark also remain
incomplete. These are the active gaps;
removed historical experiment architectures are not part of the current
runtime or comparison contract.

The accepted remaining gold-only architecture plan is deliberately
sequential:

1. compare the implemented `wide-shared-mlp-task-adapters` candidate, with
   separate residual `H -> H/2 -> H` paths for UPOS, morphology, and lemma,
   against the selected aggregation: completed and rejected;
2. compare the implemented `wide-shared-mlp-structured-morphology` decoder
   against the selected learned-last-four, Mean-pooling, `wide-shared-mlp`
   architecture without the rejected adapters: completed and selected;
3. compare the compact character-CNN branch for rare and previously unseen
   word forms against the selected structured architecture: completed and
   selected after separate Bokmål/Nynorsk and Rare/OOV evaluation.

Each stage changes one architectural variable and keeps the twelve-epoch
schedule, data, seed, optimizer, losses, Mean pooling, and evaluation policy
fixed. A later stage begins only after the previous winner is recorded. This
prevents adapter, decoder, and character-branch gains from being attributed to
the selected layer mixture and keeps the shipped-size tradeoff measurable.
All three gold-only decisions are now recorded. Teacher fine-tuning and its
separate Bokmål/Nynorsk plus Rare/OOV evaluation have completed successfully.
The first distillation run and its separate Bokmål/Nynorsk plus Rare/OOV
evaluation have completed. The distilled Student is selected as the compact
reference. A typed task-specific policy now exposes separate UPOS,
morphology, and lemma temperatures and weights without changing inference.
The first controlled candidate kept all temperatures at 1.0 and used weights
0.05/0.20/0.10 for UPOS/morphology/lemma. It has been measured and rejected:
Nynorsk Rare/OOV morphology improves, but both standards regress on broader
quality measures and Bokmål Rare/OOV morphology does not improve. Full
categorical DKD is implemented as a separate optional objective: it splits
target and non-target class knowledge for UPOS, lemma, and exclusive morphology
while preserving binary KL for multi-value morphology. Its first controlled
temperature-1.0, outer-weight-0.1 and component-weight-1.0/1.0 candidate is
measured and selected as the compact reference. It improves loss, overall
UPOS, lemma, and Rare/OOV lemma and morphology on both written standards;
localized Rare/OOV UPOS regressions remain explicit.

The rejected adapter candidate shares one adapter across all morphology
feature heads, adds 111,456 parameters at `H = 192`, and changes neither task
output spaces nor loss and decoding contracts. Its zero-initialized output
projections make all three adapter paths exact identities at initialization.
The candidate improved Nynorsk morphology micro F1 and Average Precision but
regressed Nynorsk UPOS, lemma, and macro F1 and produced broader Bokmål
regressions. It remains an export-compatible ablation option but is not part of
the selected architecture or the structured-decoder control.

The structured decoder keeps the independent morphology logits as a first
pass, concatenates soft UPOS and morphology distributions, and predicts
parallel residual corrections for every morphology feature. It has no hard
UPOS decision and no autoregressive feature order. Zero-initialized correction
heads make the candidate exactly equivalent to the selected control at
initialization. For the joint Norwegian schema it adds 23,476 parameters at
`H = 192`, approximately 94 KB in FP32. Loss, decoding, distillation, and
artifact output contracts remain unchanged; the decoder passes strict
`torch.export`. The controlled comparison selects it: Loss, Lemma,
morphology precision, recall, micro F1, and Average Precision improve on both
written standards for 105,666 additional checkpoint bytes. Small Nynorsk UPOS
and macro-F1 regressions of 0.0256 and 0.0470 percentage points remain explicit
tradeoffs. New Norwegian runs therefore default to
`wide-shared-mlp-structured-morphology`.

## Teacher and student

A teacher-student setup uses two models for two different jobs.

### Teacher

The teacher is a comparatively large pretrained Norwegian language model. It
is fine-tuned on the Prism tasks and optimized primarily for prediction
quality. Candidate teacher encoders must be compared on the development split
before one is selected; no specific external model is accepted merely because
it is popular or large.

The teacher is used during model development and training. It is not part of
the shipped Prism language package.

### Student

The student is a much smaller model designed for local inference. It learns
from two sources:

1. the correct UD annotations, called hard targets; and
2. the teacher's probability distributions or internal representations,
   called soft targets.

Soft targets contain more information than the winning class alone. For
example, a teacher may assign a verb 80% probability of past tense and 18%
probability of present tense. A gold label only says `Past`. Learning from the
distribution can teach the student which alternatives are linguistically
plausible and where the teacher is uncertain. This transfer process is called
knowledge distillation.

Only the student and the deterministic preprocessing and decoding data are
exported for applications. Distillation is valuable only if the measured
student improves over an equally sized model trained without the teacher, so
an ablation benchmark is required.

## First production task bundle

The first Norwegian model generation operates on sentences of externally
supplied tokens and returns one result per input token. It shares one
contextual encoder and uses the language-independent task-head families with
Norwegian label schemas.

The first production bundle contains:

- UPOS;
- every supported UD morphology feature observed in the pinned Norwegian
  training schema, including verb-related features such as `Tense`, `Mood`,
  `VerbForm`, and `Voice`;
- lemmatization;
- calibrated confidence for every reported prediction.

Morphology uses one classifier per feature instead of treating an entire
feature bundle as one class. Exclusive features classify `<NONE>` and their
real values with softmax and Cross-Entropy. Genuinely multi-valued features
emit independent logits only for real values, use sigmoid and Binary
Cross-Entropy, and derive `<NONE>` when no real value is active. The versioned
schema selects the contract per feature; generic training, distillation,
decoding, evaluation, and export code must preserve the same distinction.

The initial lemmatizer predicts a learned edit rule that transforms the token
form into its lemma. A later character generator is permitted only if edit-rule
coverage and unknown-word results show that it is needed. Lemma decoding must
preserve casing and Norwegian characters.

Dependency parsing is a later, explicit task. A biaffine parsing head may reuse
the contextual token representations, but it is not silently bundled into the
first morphology and lemma milestone. Tokenization, sentence segmentation,
named entities, phrases, and multiword expressions remain separate decisions.

## Model artifact and native runtime

Training remains in Python and PyTorch. A released language package is not a
Python environment or a raw training checkpoint. It is a versioned directory
containing at least:

```text
prism-no-bokmaal-<version>/
├── model-coreml.pte
├── model-xnnpack.pte
├── manifest.json
├── vocabulary.json
├── labels.json
└── LICENSES/
```

ExecuTorch `.pte` is the initial portable artifact family. Lowering and
delegation are backend-specific, so a release may contain separate artifacts
for Core ML, XNNPACK, Metal, CUDA, OpenVINO, or another measured target rather
than pretending that one optimized binary is universal. The manifest maps each
artifact to its platform, backend, precision, supported shapes, and fallback
policy. It also records the model schema version, language, tasks, tensor
contract, expected normalization, training-data provenance, model license,
quantization, and benchmark identity. Every artifact requires numerical parity
tests against the fixed PyTorch model.

ExecuTorch is the first runtime path because it provides a C++ runtime and
platform integrations for Apple and Android targets. On Apple platforms,
Prism's Swift package should wrap the runtime and select an appropriate
ExecuTorch backend such as Core ML, Metal, or XNNPACK based on measured support
and performance. The legacy ExecuTorch MPS backend is deprecated and is not a
new Prism target. A direct Core ML artifact, comparable to WhisperKit's Apple-
specific packaging, remains a valid measured alternative if it has better
coverage or performance than ExecuTorch delegation. The public Prism Swift API
must not expose ExecuTorch or Core ML types; this keeps the API stable if the
runtime implementation changes later.

The intended first package split is:

- `PrismCore`: stable token, sentence, result, confidence, and error types;
- `PrismRuntime`: internal model loading, batching, and tensor execution;
- `PrismNorwegian`: the first language profile, artifact metadata, and decoding
  behavior;
- `PrismKit`: convenient public Swift entry point.

Future Java/Kotlin and C++ libraries consume the same model manifest and tensor
contract instead of defining separate model semantics.

Future language packages follow the same boundary as `PrismNorwegian`. Adding
a language must not require copying the runtime, batching, task-head, or public
API implementations.

Applications with many learning languages must remain fully functional
without a model server or runtime network access. Every language exposed by an
installation must have its versioned artifacts in local storage. The runtime
loads only the currently used artifacts into memory and may release inactive
ones, but it must not fetch them. Closely related written standards may share
one artifact when separate evaluation proves that sharing is beneficial.
Broader multilingual backbones, compact language-specific adapters,
distillation, quantization, and installation packaging must be compared before
Prism scales toward dozens of locally available languages.

## Document inference contract

LexKeep may supply documents with roughly 200 sentences and 6,000 tokens.
Prism must preserve the caller's sentence and token boundaries and return
results in the same order. It must batch compatible sentences instead of
treating the entire document as one recurrent or Transformer sequence.

The first production performance gate is measured on a recorded Apple Silicon
reference machine with a 6,000-token, 200-sentence fixture:

- warm end-to-end inference for UPOS, morphology, lemmas, and confidence:
  median at most 1.0 second and p95 at most 1.5 seconds;
- cold load plus first inference: at most 3.0 seconds;
- peak additional memory: at most 250 MiB;
- distributed, quantized Norwegian student package: at most 100 MiB.

These are release gates, not claims about unimplemented code. Measurements
must record the hardware, operating system, runtime version, backend, thread
configuration, batch policy, model format, quantization, and whether model
loading is included. If a backend cannot meet the gate, Prism must optimize,
distill, quantize, or choose a smaller student rather than hiding latency in an
application layer.

## Quality and comparison contract

Model selection uses the pinned training and development splits. The test split
is evaluated only after the architecture, hyperparameters, confidence method,
and export configuration are fixed.

At minimum, every candidate report includes:

- UPOS accuracy and per-class precision, recall, F1, and support;
- each morphology feature's overall accuracy, annotated-token accuracy, and
  per-value precision, recall, F1, and support;
- complete morphology-bundle exact match;
- lemma accuracy overall, on known tokens, and on unknown tokens;
- confidence calibration, including expected calibration error and selective
  accuracy at documented abstention thresholds;
- model size, peak memory, cold latency, warm latency, and tokens per second.

The distilled student must be compared with the same student trained without
distillation, its teacher, and an independently reproduced external pipeline
such as UDPipe on compatible data and input conditions. Gold-token and
raw-text evaluations must never be mixed. Prism can claim to match or beat
another system only when the dataset revision, splits, tokenization condition,
tasks, and metrics are genuinely comparable.

## Repository structure

Production Python code is organized by stable concerns under data/schema,
language profiles, model components, training, evaluation, export, and
artifact loading. Historical recurrent and dictionary experiments were
removed after the Transformer student surpassed their documented scope and
became the gold-only distillation reference. New work extends the shared typed
pipeline rather than creating a parallel experiment namespace.
