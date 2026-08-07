# Prism model and runtime strategy

Status: active architectural direction with selected shared Norwegian student
Last updated: 2026-07-24

## Purpose

Prism should provide serious, locally runnable language models rather than a
collection of isolated training demonstrations. The first production target
remains Norwegian Bokmål, externally tokenized by the host application.
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

## English expansion (Ettin backbone)

English is the first non-Norwegian language profile. It is a deliberate test
of the language-independent core: the model heads, training, distillation,
silver pipeline, calibration, export, and native runtimes stay **unchanged**,
and only the language profile — backbone family, tokenizer, treebank, and (to
come) silver adapters — is replaced. `python/src/prism/languages/norwegian/`
is not touched; the English package sits beside it under
`python/src/prism/languages/english/`.

**Why the backbone family had to change.** NorBERT4 is not the classic
LTG-BERT; it is the GPT-BERT architecture (`GptBertForMaskedLM`: RoPE,
local–global attention, loaded through `trust_remote_code`). There is no
modern, large-corpus **English** model in that same architecture with the
size range Prism needs: same-architecture English weights exist only at
BabyLM scale (`ltg/gpt-bert-babylm-*`, ~100M words, base size only — no
compact student, no large teacher). Keeping GPT-BERT literally would have
required pretraining English encoders from scratch on a web corpus
(GPU-cluster scale, off the MPS development path). The profile therefore
adopts the closest modern family that natively spans both required sizes.

**Backbone decision.** Three options were evaluated against the two size
targets (student ≈ `norbert4-xsmall` 17M, teacher ≈ `norbert4-large` 360M):

| Option | Student / teacher | Verdict |
| --- | --- | --- |
| **Ettin encoder suite** (ModernBERT lineage, Weller et al. 2025) | `jhu-clsp/ettin-encoder-17m` (16.80M) / `jhu-clsp/ettin-encoder-400m` | **Selected** |
| Classic BERT (Turc et al. 2019 miniatures + BERT-large-WWM) | `google/bert_uncased_L-12_H-256_A-4` (~17.5M) / `bert-large-*-whole-word-masking` (~335M) | Fallback only |
| GPT-BERT BabyLM | `ltg/gpt-bert-babylm-base` | Rejected |

Ettin is objectively strongest on every quality-relevant axis: ~2T open
tokens versus BERT's ~3.3B (2018) and BabyLM's ~100M; a modern architecture
(RoPE, local–global attention, GeGLU) conceptually close to GPT-BERT rather
than a generation backwards; and, decisively for distillation, a **paired
suite** — student and teacher share one tokenizer, recipe, and data, so
soft-label alignment is clean. The proposed classic-BERT pairing additionally
mixed an uncased WordPiece-30522 student with a cased WordPiece-28996 teacher.
MIT-licensed with fully open data. Ettin's one theoretical disadvantage —
ModernBERT lowering to ExecuTorch was unproven — was the reason for the export
spike below; it is retired. The classic-BERT pairing is kept documented as a
zero-export-risk fallback, not adopted. The teacher starts at 400M (like-for-like
with `norbert4-large`); `ettin-encoder-1b` is a registered upgrade path.

**Architecture impact is contained.** Unlike NorBERT4, Ettin/ModernBERT is a
first-class `transformers` architecture, so `trust_remote_code` is off. Two
settings are pinned for a portable, deterministic export graph:
`attn_implementation="eager"` (avoids SDPA/flash and unpadding paths) and
`reference_compile=False` (no internal `torch.compile` during export capture or
MPS training). These are realized as a backward-compatible extension to
`PretrainedBackboneSpec` — new optional `attention_implementation` and
`config_overrides` fields whose defaults preserve the exact prior behaviour for
the Norwegian specs — so the shared loader stays generic and Norwegian is
byte-for-byte unaffected.

**Export spike (risk retired).** `ettin-encoder-17m`, wrapped in Prism's own
`BackboneExportAdapter`, lowers cleanly through the existing production path
`lower_to_executorch_xnnpack` (`torch.export(strict=True)` →
`to_edge_transform_and_lower` → XNNPACK → `to_executorch`) with fp32 parity of
**2.7·10⁻⁵** against eager, and program-data separation produces a small `.pte`
plus a shared `model.ptd` — exactly the structure of the shipped
`quantization: none` Norwegian artifact. The NorBERT4-specific
`fold_scaled_linear_parametrizations` pass is a correct no-op on ModernBERT.
The only open item is int8 PT2E quantization (a RoPE/mask advanced-indexing
edge in the calibration run); it is not part of the shipped configuration and
is deferred.

**Data.** Gold: `UD_English-EWT` (CC-BY-SA-4.0, ~254k tokens — the largest
English UD treebank), pinned by revision; `UD_English-GUM` is excluded because
it is CC BY-NC-SA 4.0 (non-commercial). English has no written-standard split,
so there is a single `en` profile rather than the `nb`/`nn`/`no` triple.
Silver (planned, mirroring the Norwegian adapters): Project Gutenberg
(public domain, the `nbdigital` analogue) and an English Wikipedia dump
(CC BY-SA 4.0, the `wikipedia` analogue).

Primary references:

- [Ettin suite](https://huggingface.co/blog/ettin) · [paper](https://arxiv.org/abs/2507.11412)
- [ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) · [ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m)
- [UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT)

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

Host applications may supply documents with roughly 200 sentences and
6,000 tokens.
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

For UDPipe-compatible gold-token comparisons, Prism additionally reports the
official word-alignment score objects for `UPOS`, complete universal `UFeats`,
and `Lemmas`: correct/gold/system/aligned counts, precision, recall, F1, and
aligned accuracy. Prism must not label its per-feature morphology accuracy or
micro-F1 as UDPipe `UFeats`; UDPipe requires the complete universal feature
bundle of one word to match exactly. `XPOS`, `AllTags`, UAS, LAS, MLAS, BLEX,
Words, and Sentences remain unavailable until Prism implements the
corresponding outputs or raw-text tasks.

The distilled student must be compared with the same student trained without
distillation, its teacher, and an independently reproduced external pipeline
such as UDPipe on compatible data and input conditions. Gold-token and
raw-text evaluations must never be mixed. Prism can claim to match or beat
another system only when the dataset revision, splits, tokenization condition,
tasks, and metrics are genuinely comparable.

### Closing the complete-bundle UFeats gap

The first UDPipe comparison localizes most of the remaining morphology gap in
`Gender`, followed by `Number` and `Definite`; it does not justify a broad
backbone-size increase. Prism will test the following interventions in order:

1. analytically remove a controlled share of the prior shift introduced by
   morphology class weighting before decoding;
2. represent treebank annotation conventions as language-profile output
   policies around one canonical shared Norwegian prediction space;
3. add a compact bundle-aware candidate scorer only if independent feature
   decoding remains the measured bottleneck;
4. test a bounded local agreement refiner only where the error audit shows
   cross-token context is available but the correct bundle is already in the
   candidate inventory;
5. add licensed Norwegian silver data only after the teacher, canonical target
   space, and output policies are frozen, so pseudo-labels cannot amplify an
   annotation-contract mistake.

This work does not make UDPipe's architecture Prism's target architecture.
UDPipe is an external quality reference, not a teacher, decoder specification,
or source of labels for model selection. Prism deliberately retains one
schema-driven classifier per morphology feature as its primary public
contract. This supports per-feature confidence, feature-level error analysis,
genuinely multi-valued features, and combinations that were not observed as
complete training bundles. The structured decoder, bundle reranker, and direct
bundle loss are coherence layers around those predictions; they must not turn
the model into a closed whole-tag lookup merely to improve one benchmark
column.

Complete-bundle `UFeats` and feature-level quality answer different questions.
`UFeats` is strict conjunction accuracy: one wrong value makes the entire word
incorrect. It therefore measures whether Prism emits a completely consistent
UD analysis, but it does not show which system is better for every individual
feature. Every future UDPipe comparison must consequently report both:

- official complete-bundle `UFeats` on identically tokenized data;
- for every shared feature, overall and annotated-token exact accuracy plus
  per-value precision, recall, F1, and support;
- Rare/OOV feature summaries for Prism, with the frequency classes derived
  only from its training data;
- explicit convention mappings separately from canonical model quality.

The currently documented direct feature comparison establishes that UDPipe is
better on the three largest deficit features, `Gender`, `Number`, and
`Definite`, for the measured historical DKD Student. It does not establish
that UDPipe is better on every other feature, and it must not be generalized
that way. A complete all-feature comparison against the selected checkpoint is
therefore a required diagnostic before feature-specific objectives are
changed.

The first intervention is a selected output policy, not a trained model
change. For class weight `w`, raw logit `z`, and strength `a`, decoding uses
`z_corrected = z - a * log(w)`. The checkpointed training weights are the only
weight source. The fixed Development grid of `0.0/0.25/0.5/0.75/1.0` selects
full correction at `a = 1.0` for the shared Norwegian release policy. It closes
44.4% of the Bokmål and 45.2% of the Nynorsk Development UFeats gap to UDPipe;
both test splits remain untouched.

The release contract stores the resolved per-feature offset vectors
`-log(w)`, the selected strength, their schema association, and provenance in
the versioned artifact. The tensor-only export adapters now register the
resolved `strength * log(w)` vectors as fixed model buffers and subtract them
inside the exported graph after the model's raw morphology logits. Strict
export parity covers the selected character-aware path, so native runtimes do
not have to reimplement or remember the correction. The CLI stays at zero by
default for backward compatibility with checkpoints that do not contain
training weights. Manifest serialization and production artifact construction
must still record and automatically select the policy.

Treebank-specific output policies must not silently redefine Prism's public
morphology. They exist to map a documented canonical representation to an
external annotation convention. Mixed Norwegian input continues to use one
shared model and canonical output unless a caller explicitly requests a
compatible external convention.

The production export decision is profile-specific rather than model-specific.
One shared Norwegian neural artifact serves `nb`, `nn`, and `no`; the manifest
records versioned output profiles and the runtime selects one explicitly.
`nb` and mixed or unspecified `no` use canonical morphology. An explicitly
requested external Nynorsk UD output selects the Nynorsk treebank policy. The
policy must not be baked unconditionally into the shared graph. Separate
Bokmål and Nynorsk package wrappers may reuse the same model bytes and differ
only in manifest and default output profile. Python, Swift, and C++ parity
tests must cover every exported profile. Manifest serialization and native
profile selection remain implementation work.

The first reversible implementation is exposed as
`--ud-morphology-policy treebank`. It maps `Gender=Fem,Masc` to `Gender=Com`
for adjectives and determiners. The Nynorsk policy additionally removes
`Number=Sing` and `Definite=Def`, values that its pinned training split does
not normally express. This transformation affects only official UFeats
scoring; raw per-label and Rare/OOV reports remain canonical. It is an
evaluation candidate until separately measured on Bokmål and Nynorsk
Development with the already selected logit correction.

With the selected Bundle-32 Student, Bokmål canonical UFeats is 96.0076%; its
treebank policy changes six bundles, improves none, regresses one, and lowers
the score marginally to 96.0048%. It is therefore not selected for Bokmål.
Nynorsk improves from 93.6384% canonical UFeats to 96.1920% and exceeds the
reproduced UDPipe result by 0.5312 points. Its per-rule audit finds 618, 138,
and 42 sequentially improved bundles and zero regressions. Those 798
improvements exactly account for the 2.5536-point aggregate gain. The mapping
therefore explains the former Nynorsk gap as an annotation-contract mismatch,
not a need for a second model, and remains restricted to explicitly requested
external Nynorsk output.

The candidate-space gate for intervention 3 is also complete. A joint
training-only inventory contains 256 complete bundles and at most 74 bundles
for one UPOS. With gold UPOS, the top 16 frequency-ranked candidates cover
93.0986% of annotated Bokmål and 90.3684% of annotated Nynorsk Development
tokens; top 32 covers 98.7736% and 94.8125%. Therefore the first compact
bundle-aware scorer may consider up to 32 candidates per UPOS. It must combine
bundle evidence with the existing independent feature logits and retain the
independent decoder as a fallback for unseen bundles and predicted-UPOS
errors. Oracle coverage is only an architectural feasibility bound, not a
quality result, and does not authorize evaluation on the test splits.

The Top-32 scorer is implemented as an optional checkpointed model component.
It never consumes gold UPOS during training-time evaluation or inference:
soft predicted UPOS, independent feature likelihoods, and a learned
token-to-candidate projection jointly score the fixed training-derived
candidates. Candidate marginals enter each feature head through a learned
residual gate rather than replacing the independent logits. This preserves a
path for unseen combinations and avoids a hard predicted-UPOS cascade. The
evaluation CLI can disable the component on the same checkpoint, and the
enabled path has strict export parity. Canonical Bokmål and Nynorsk,
per-label, Rare/OOV, and complete-bundle gains selected it as the current
Student standard; UDPipe ranking was not the acceptance criterion.

The fourth intervention was implemented and measured as an optional agreement
refiner. It runs after the structured morphology decoder and Top-32 scorer,
projects the token representation, soft UPOS, and all current morphology
probabilities into a 64-dimensional bottleneck, and attends only to valid
neighbors within three positions. Attention to the current token is excluded
so the ablation specifically measures additional sentence context. Gated
residual heads can alter only `Definite`, `Gender`, and `Number`; they start as
a no-op and add 29,707 parameters, about 119 KB in raw FP32. The component uses
no gold or comparison-system signal at inference, is checkpointed, can be
disabled for diagnosis, and has strict export coverage. The separately trained
run is rejected: Bokmål UFeats, all three targeted annotated-feature
accuracies, and Rare/OOV morphology regress. Nynorsk improves on UFeats and
Rare morphology but regresses on OOV morphology, so the shared Norwegian
acceptance contract is not met. This result argues against generic soft local
agreement as the next production change; Bundle-32 remains selected. Neither
UDPipe imitation nor a same-checkpoint disable pass was used as selection
evidence.

### Licensed silver-data pipeline

The first silver source is the CC0 NBdigital Bokmål corpus
`oai:nb.no:sbr-43` from the National Library of Norway's Språkbanken. It was
chosen instead of scraping a general ebook catalogue because it provides one
documented archive, machine-readable provenance, and a uniform reuse license.
The archive's existing Oslo-Bergen analyses are deliberately ignored: they
are neither Prism's canonical label space nor the accepted Teacher.

The preferred Nynorsk running-text companion is Språkbanken resource
[`oai:nb.no:sbr-60`](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-60/).
It is CC0, contains 50,000 OCR-derived municipal documents and approximately
127 million words, about 88.5 million of them classified as Nynorsk. Its
language-classified JSON pages make it a substantially cleaner first Nynorsk
silver source than a mixed ebook scrape. It is not yet downloaded or prepared;
it needs a separate source adapter plus the same deduplication, OCR, length,
and all-UD-split overlap gates as the Bokmål source.

The CC0
[`oai:nb.no:sbr-65`](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-65/)
Nynorsk pronunciation lexicon is tracked separately because its inflected
forms, lemmas, and lexical features may provide targeted lexical Gender/OOV
supervision, but it has no sentence context and is not a silver-text
replacement. The Nynorsk Norsk ordbank catalogue entry is CC BY rather than
public domain and therefore remains outside the CC0 pilot pending a separate
provenance and redistribution decision.

Silver training is split into three independently validated artifacts:

1. **Source preparation.** `prepare_silver_corpus` streams the archive, retains
   supplied words and explicit sentence boundaries, filters low-confidence
   OCR and oversized sentences, deduplicates normalized token sequences, and
   excludes fingerprints from every Norwegian UD split.
2. **Offline Teacher labeling.** The frozen accepted Teacher will label the
   prepared sentences once. The artifact must bind hard pseudo-labels and
   per-task confidence to the source-manifest hash, Teacher-checkpoint hash,
   schema version, and resolved morphology output policy. It must not depend
   on an online Teacher during each Student epoch.
3. **Controlled Gold/Silver training.** Gold remains the authoritative target.
   Silver losses are confidence-filtered or confidence-weighted and receive an
   explicit global mixture weight. Gold batches for Bokmål and Nynorsk keep
   their documented sampling policy. Model selection continues to use only
   untouched gold Development splits and requires gains on both written
   standards.

The first stage is implemented. The other two are intentionally not hidden
inside the source importer: that separation keeps corpus licensing,
pseudo-label provenance, and optimization policy independently auditable. No
silver benchmark exists until a fixed pseudo-label policy and a controlled
gold-only comparison have both been evaluated.

The pre-silver quality gate confirms that the corrected historical
character-aware Teacher is stronger than the compact canonical Student on
UPOS and UFeats for both written standards and on Nynorsk Lemmas, but it
predates the selected Top-32 bundle reranker. Since offline labeling of the
prepared 50-million-token corpus will cost much more than one gold-only
Teacher run, the label source must first be retrained with the final selected
Bundle-32 architecture. The rejected agreement refiner remains excluded.
Development selection and separate Bokmål/Nynorsk acceptance stay unchanged;
this is architecture alignment, not permission to tune on silver data.

The architecture-matched Base result improves corrected exact UFeats and Rare
morphology on both written standards, but slightly regresses Lemmas on both
and Bokmål UPOS/OOV morphology. It is therefore the primary morphology
control, while the historical corrected Base Teacher remains an explicit
agreement control.

NorBERT4-large is deferred until the task objectives and selection criterion
are audited. Training now supports a direct complete-bundle objective in
addition to UPOS, mean-per-feature morphology, and flat lemma-rule losses.
The first compact run closed 0.7314 points of the Bokmål UFeats gap and 0.8288
points of the Nynorsk gap, but it also caused a repeatable lemma tradeoff,
including a 0.7838-point Bokmål OOV-lemma regression. This demonstrates that
objective alignment, rather than backbone capacity alone, remains actionable.
The lemma head still remains a flat edit-rule classifier without soft UPOS or
morphology conditioning.

The accepted order before a larger Teacher is now:

1. keep the selected twelve-epoch morphology-scoped compact Student;
   `residual-only` remains the protected-gradient control and the completed
   30-epoch isolated schedule remains rejected;
2. retain the completed per-feature Prism-versus-UDPipe Development report:
   both standards concentrate the residual error in `Gender`, `Number`, and
   `Definite`, while Prism already leads several other features and Nynorsk
   additionally exposes a separate annotation-convention component;
3. retain the completed `morphology` gradient-scope result: it trains
   morphology-specific heads and decoder parameters while keeping Backbone,
   shared representation, UPOS evidence, and lemma representation protected,
   and the joint Bokmål/Nynorsk gate selected it for its complete-split and
   OOV UFeats gains;
4. retain the completed two-seed frozen-head probe and full-training gate:
   the shared post-fusion residual `H -> 2H -> H` morphology MLP is selected
   after gaining 226 complete UFeats bundles and removing 204 Gender errors
   across Bokmål/Nynorsk; the small combined OOV and lemma regressions remain
   explicit guardrails, while the larger feature-specific MLP stays rejected;
5. use `shared-mlp` for new Norwegian training runs while keeping explicit
   `identity` reproduction and checkpoint-metadata fallback for older
   artifacts;
6. retain the completed task-interaction audit: ranking dominates covered
   errors, general gradient conflict is not established, and Nynorsk has an
   additional candidate-coverage problem;
7. reject the completed nonlinear `compositional-mlp` scorer as the final
   output path: it improves internal Top-1 ranking but regresses final UFeats,
   Rare/OOV morphology, and Lemmas under the current static fusion;
8. retain the completed frozen token- and feature-dependent probability-fusion
   gate as a rejected diagnostic after it failed the complete
   Bokmål/Nynorsk and Rare/OOV gate;
9. retain the completed candidate-coverage decomposition: Top-64 removes all
   pruning misses on both standards, while never-seen bundles affect only one
   Bokmål and 150 Nynorsk Development tokens;
10. retain the completed controlled Top-64 reranker ablation as rejected: it
    removes pruning misses but regresses Bokmål UFeats and Rare/OOV quality
    for only a small Nynorsk gain; keep Top-32 selected and defer bounded
    compositional generation;
11. defer exact-bundle supervision on the final post-fusion probabilities as
    a documented architecture option and implement the offline,
    provenance-carrying Teacher-label artifact first;
12. run a deterministic confidence-filtered pilot on the already prepared
    Bokmål source, then add the CC0 Nynorsk municipal JSON adapter and a
    balanced two-standard silver sampling policy;
13. accept silver only through a matched Gold-only versus Gold/Silver Student
    comparison covering both standards, every feature, Rare/OOV, UPOS, and
    Lemmas;
14. add a frozen lemma near-miss reranker only after morphology stabilizes;
    provide it soft UPOS/morphology context only if the audit proves that this
    context resolves errors beyond character/edit-rule evidence;
15. reconsider final-output supervision or NorBERT4-large only after the
    silver-data control exposes a remaining structured-decoding or capacity
    limit.

This order changes the actual learned model rather than post-processing a
benchmark. Any candidate must improve the canonical Bokmål and Nynorsk gold
Development reports and preserve untouched test splits before it may label
silver data.

The task-interaction audit is now implemented and completed on the selected shared-MLP
checkpoint. It attributes 930 of 1,223 Bokmål complete-bundle errors and 880
of 1,851 Nynorsk errors to candidate ranking, versus only 72 and 782 missing
gold candidates. Among covered errors, the gold bundle is in the first two
candidates for 74.11% of Bokmål and 71.19% of Nynorsk cases. This supplied
sufficient evidence for the completed controlled scorer ablation.

The new `compositional-mlp` scorer replaces only the learned linear residual.
It composes candidate vectors from schema-derived UPOS and feature-label
embeddings, computes a nonlinear token query, and scores their compatibility.
Its final query projection starts at zero, preserving the existing candidate
evidence exactly at initialization. The old `linear` scorer remains the
training default and the fallback for checkpoint metadata that predates the
field. The current Norwegian reranker grows from 35,723 to 89,298 parameters,
and both variants pass strict export.

The Bokmål end-to-end gate rejects this scorer as the new default. It raises
UPOS by 0.0330 points but lowers UFeats by 0.0412, Lemmas by 0.0660, Rare
UFeats by 0.3809, and OOV UFeats by 0.4987 points. The internal audit explains
the apparent contradiction: ranking errors fall by 59 and covered-token
candidate Top-1 rises by 0.1385 points, while refinement errors rise by 56.
The learned nonlinear compatibility is useful signal, but the current static
residual logit fusion does not use it reliably.

The completed frozen adaptive probability-fusion probe was rejected and
followed by candidate-coverage decomposition rather than another Backbone or
scorer expansion. The probe learned only a compact gate between
independent feature probabilities and marginalized bundle probabilities,
using frozen token context, feature identity, entropy, margins, and
disagreement. It never used Development labels or UDPipe predictions.

Despite falling training loss, exact UFeats regressed by 0.9733/0.3488 points
on Bokmål/Nynorsk and Rare/OOV regressed on both standards. Most learned gates
were near one, so the convex mixer nearly collapsed onto the bundle expert
and failed to preserve the selected static residual fusion's generalization.
The artifact remains diagnostic; it is not integrated into full training,
checkpoint metadata, or export. Its executable probe paths were removed after
documenting the result, and the production checkpoint remains unchanged.

The completed read-only candidate decomposition separates Top-K pruning
from genuinely unseen bundles. Top-32 covers 99.2934% of Bokmål but only
96.9664% of Nynorsk Development. Top-64 reaches the complete inventory's
99.9973%/99.5200% coverage and removes 256/798 pruned-token misses, including
11/96 OOV misses. The full inventory still cannot cover one/150
Bokmål/Nynorsk tokens.

The subsequent matched Top-64 training result rejects candidate expansion as
the shared default. Against its matched Top-32 control, Bokmål UFeats falls
0.0413 points, Rare falls 0.2857, and OOV falls 0.1069. Nynorsk UFeats gains
0.0832 points and OOV gains 0.1972, but UPOS loses 0.0192. Across both full
splits the net UFeats gain is only 11 tokens despite runtime candidate
coverage rising from 98.2180% to 99.7767%. This fails the fixed two-standard
and Rare/OOV gate. Top-32 remains the selected contract. Post-fusion
exact-bundle supervision remains a documented architecture option, but the
immediate implementation boundary is now the offline Teacher-label artifact
and a controlled silver-data pilot. The temporary Top-64 CLI/test surface and
expanded read-only coverage decomposition were removed after the result was
recorded, leaving no rejected production path.

Gradient-conflict mitigation is not selected from this audit. Average gradient cosines remain
positive across both standards; only Nynorsk UPOS-versus-Lemma gradients in
the shared projection show a majority-negative sample (9 of 16 batches).
This does not establish a general morphology-versus-lemma conflict and must
not be combined with the scorer run. Soft lemma context also remains gated:
the lemma-rank audit shows large OOV Top-2 headroom, but does not yet prove
that UPOS/morphology context, rather than character or edit-rule evidence,
resolves those errors.

A second parallel whole-`UFeats` classifier is technically possible, but it
would duplicate an existing responsibility. The selected Bundle-32 reranker
already acts as a complete-bundle expert: it scores whole training-derived
UPOS/morphology candidates and marginalizes their distribution back into the
individual feature logits. The direct bundle objective already supplies the
corresponding whole-bundle supervision. Adding another flat classifier over
the same bundles would split scarce bundle examples between two competing
paths, repeat the same closed-inventory limitation, and make it less clear
which path owns final calibration.

The completed scorer ablation therefore changed this existing UFeats path
instead of adding a duplicate head. Its token side uses a small nonlinear
query projection; its candidate side composes a representation from
schema-derived UPOS and feature-value labels; their compatibility contributes
a zero-initialized residual energy to the existing independent-head evidence.
The candidate distribution is still marginalized back into each feature, and
the independent decoder remains the fallback for combinations outside the
inventory. This kept the change language-independent and tested the measured
candidate-ranking hypothesis directly.

The direct bundle objective is an auxiliary term, not a replacement for the
18 feature objectives. It marginalizes identical complete morphology bundles
across UPOS-specific candidate entries, so it matches UFeats without making
gold UPOS part of that target. Candidate coverage and auxiliary loss are
reported per epoch. The first controlled Student candidate used weight `0.1`;
weight `0` reproduces the existing Bundle-32 training objective. It selected
epoch 12 and validated the morphology objective on both standards, but failed
the all-task gate because Lemmas regressed. Restricting the same auxiliary
gradient to the bundle residual scorer then improved canonical UFeats, Lemmas,
Rare/OOV morphology, and OOV lemma over Bundle-32 on both standards, with only
single-digit error-count UPOS and Nynorsk Rare-lemma trades. It served as the
protected-gradient reference, and its rejected 30-epoch extension confirmed
that schedule length was not the bottleneck. The subsequent `morphology` scope
widened only through morphology-owned parameters. It gains 73 complete-split
UFeats predictions and 37 OOV UFeats predictions across both standards versus
`residual-only`, while five of six complete-split task metrics improve and
inference remains identical. That twelve-epoch checkpoint is now the selected
compact reference; the measured Rare and small OOV UPOS/Lemma trades remain
explicit guardrails for the next intervention.

The production CLI follows this selection: a positive direct-bundle loss with
no explicit gradient scope resolves to `morphology`; zero bundle-loss weight
continues to resolve to `full` because the scope is then inactive. Explicit
scope values remain required in benchmark commands whenever the scope itself
is the controlled variable.

## Repository structure

Production Python code is organized by stable concerns under data/schema,
language profiles, model components, training, evaluation, export, and
artifact loading. Historical recurrent and dictionary experiments were
removed after the Transformer student surpassed their documented scope and
became the gold-only distillation reference. New work extends the shared typed
pipeline rather than creating a parallel experiment namespace.
