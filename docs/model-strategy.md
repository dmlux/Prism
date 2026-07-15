# Prism model and runtime strategy

Status: accepted architectural direction, implementation not yet started  
Date: 2026-07-15

## Purpose

Prism should provide serious, locally runnable language models rather than a
collection of isolated training demonstrations. The first production target
remains Norwegian Bokmål, externally tokenized by a caller such as LexKeep.
The architecture must later allow separate language packages behind the same
public API.

This document defines the direction for the next model generation. It does not
claim that the new model already exists or outperforms the recurrent baseline.
Every such claim requires a recorded benchmark on the pinned data splits.

## Why the current model is not the final architecture

The current word-and-character BiLSTM is a useful baseline. It learns directly
from the annotated Universal Dependencies training split and jointly predicts
UPOS and the single morphology feature `Number`. It is small, fast, and has
already demonstrated that character information helps unknown Norwegian
words.

Its limitations are structural:

- it receives no broad Norwegian language knowledge learned from unannotated
  text;
- morphology is hard-coded to one feature head;
- combined feature values are represented as rare monolithic classes;
- there is no lemmatization objective;
- there is no stable export artifact or native runtime contract;
- evaluation currently measures sentences from a treebank, not complete
  LexKeep-sized documents through a production API.

The recurrent model remains a reproducible baseline until a replacement has
been shown to be both better and fast enough. Existing results must not be
deleted merely because a newer architecture is being developed.

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

The first new model generation operates on sentences of externally supplied
tokens and returns one result per input token. It shares one contextual encoder
and uses task-specific output heads.

The first production bundle contains:

- UPOS;
- every supported UD morphology feature observed in the pinned Norwegian
  training schema, including verb-related features such as `Tense`, `Mood`,
  `VerbForm`, and `Voice`;
- lemmatization;
- calibrated confidence for every reported prediction.

Morphology uses one classifier per feature instead of treating an entire
feature bundle as one class. Each head has an explicit `<NONE>` value for a
token on which that feature does not apply. Multi-valued annotations require a
documented representation and tests; they must not silently become a single
rare label as in the current prototype.

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
├── model.pte
├── manifest.json
├── vocabulary.json
├── labels.json
└── LICENSES/
```

`model.pte` is the initial target format produced for ExecuTorch. The manifest
records the model schema version, language, tasks, tensor contract, expected
normalization, maximum supported shapes, training-data provenance, model
license, quantization, and benchmark identity. Export must include numerical
parity tests between PyTorch and the exported model.

ExecuTorch is the first runtime path because it provides a C++ runtime and
platform integrations for Apple and Android targets. On Apple platforms,
Prism's Swift package should wrap the runtime and select an appropriate
ExecuTorch backend such as Core ML, MPS, or XNNPACK based on measured support
and performance. The public Prism Swift API must not expose ExecuTorch types;
this keeps the API stable if the runtime implementation changes later.

The intended package split is:

- `PrismCore`: stable token, sentence, result, confidence, and error types;
- `PrismRuntime`: internal model loading, batching, and tensor execution;
- `PrismNorwegian`: Norwegian artifact metadata and decoding behavior;
- `PrismKit`: convenient public Swift entry point.

Future Java/Kotlin and C++ libraries consume the same model manifest and tensor
contract instead of defining separate model semantics.

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

The student must be compared with the current BiLSTM baseline, the same student
trained without distillation, its teacher, and an independently reproduced
external pipeline such as UDPipe on compatible data and input conditions.
Gold-token and raw-text evaluations must never be mixed. Prism can claim to
match or beat another system only when the dataset revision, splits,
tokenization condition, tasks, and metrics are genuinely comparable.

## Repository transition

The completed recurrent implementation is isolated under
`prism.baselines.recurrent`, including its tensor datasets, vocabularies,
models, training, evaluation, inference, and CLI implementations. The original
top-level model and command modules have been removed; baseline commands use
the explicit `prism.baselines.recurrent.cli` namespace. Recurrent checkpoint
reconstruction continues to work. The dictionary comparison lives separately
under `prism.baselines.dictionary`.

The next-generation path can now introduce cleanly separated data/schema, task
heads, teacher experiments, student models, training, evaluation, export, and
artifact tooling without extending the recurrent baseline modules.

No old training entry point or checkpoint format should be removed before:

- its documented benchmark remains reproducible;
- the replacement has passed the quality and runtime gates;
- any required checkpoint migration is documented;
- `README.md`, `docs/PROJECT_STATUS.md`, and `docs/benchmarks.md` reflect the
  transition.
