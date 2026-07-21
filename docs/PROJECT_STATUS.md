# Prism project status

Last updated: 2026-07-21

## Product direction

Prism is a modular, open-source NLP toolkit for fast, local, privacy-friendly
linguistic analysis. The first production family is Norwegian, with separate
Bokmål (`nb`) and Nynorsk (`nn`) language profiles behind shared model,
training, evaluation, export, artifact, and native API contracts.

The production target is one shared Norwegian teacher and one shared compact
Norwegian student for Bokmål, Nynorsk, and mixed written input. The separate
profiles preserve dataset provenance, schemas, sampling, and per-standard
evaluation; they do not require separate shipped model weights. A shared model
must still match or improve the separately trained references on both
development splits.

Prism is intended to support many future languages while preserving LexKeep's
offline-first contract. There is no model server: every language advertised by
an installation must already be available in local storage and inference must
never require a network connection. The runtime may load only the currently
used local artifacts into memory. Closely related written standards may share
a measured model family; broader multilingual sharing, distillation,
quantization, and packaging must keep the complete local installation
practical without weakening per-language quality reporting.

The shipped model must remain compact enough for complete LexKeep documents
of roughly 200 sentences and 6,000 tokens. A high-capacity teacher is a
development dependency only; the released runtime contains the measured
student.

The first model bundle covers:

- UPOS;
- the supported UD morphology features;
- lemma edit-rule generation;
- calibrated confidence.

Dependency parsing, raw-text tokenization, sentence segmentation, named
entities, phrases, and multiword expressions remain explicit later tasks.

## Supported environment

- Python 3.12
- repository-local `.venv`
- distribution name: `prism-nlp`
- Python package: `prism`
- primary development device: Apple MPS
- source license: Apache License 2.0

Development setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev]'
```

## Pinned data

### Norwegian Bokmål

- repository: `UniversalDependencies/UD_Norwegian-Bokmaal`
- commit: `396d11f0c2bd290a2a2711015c04ac25bc3dcc06`
- license: CC BY-SA 4.0
- training sentences/tokens: 15,696 / 243,886
- development sentences/tokens: 2,409 / 36,369
- test split: untouched by current Transformer model development

### Norwegian Nynorsk

- repository: `UniversalDependencies/UD_Norwegian-Nynorsk`
- commit: `aaeb9d90c748c2bd9e272f180b599484f9f05ac6`
- license: CC BY-SA 4.0
- local path: `data/raw/UD_Norwegian-Nynorsk`
- status: downloaded and pinned; language profile and shared Norwegian UD
  adapter are implemented
- target: standard written-language treebank, not NynorskLIA

Both local dataset repositories use the local branch `prism-pinned` at their
documented commit. `data/raw/` is ignored by the Prism repository.

## Implemented architecture

The shared token pipeline is language-independent:

```text
externally supplied tokens + spacing
                |
language tokenizer and normalization
                |
subword IDs + token alignment
                |
replaceable language backbone
                |
contextual token representations
        +-------+----------------+
        |                        |
      UPOS              morphology feature heads
        |
    lemma edit-rule head
```

The Norwegian Bokmål profile selects `ltg/norbert4-xsmall`. Generic Prism code
does not import or branch on NorBERT4. The reusable heads receive their output
dimensions from the selected language schema.

Implemented shared components include:

- typed pretrained-backbone and language-profile contracts;
- separate Bokmål and Nynorsk profiles over the shared NorBERT4-xsmall
  backbone specification;
- one shared Norwegian UD normalizer, sentence encoder, and schema builder;
- tokenizer loading and whitespace-preserving pretokenized input;
- subword-to-token alignment;
- contextual backbone execution;
- shared normalization before task heads;
- UPOS, per-feature morphology, and lemma edit-rule heads;
- schema-aware targets, losses, decoding, and metrics;
- a hybrid morphology contract: categorical softmax/Cross-Entropy for
  exclusive features and sigmoid/Binary Cross-Entropy over real values for
  genuinely multi-valued features;
- derived `<NONE>` output for multi-valued features instead of a redundant
  trainable `<NONE>` logit;
- MPS-aware batches and device transfer;
- differential AdamW, gradient clipping, and warmup/decay scheduling;
- reproducible shuffled multi-epoch training;
- atomic best-checkpoint replacement;
- JSON-compatible schema and checkpoint metadata;
- versioned ExecuTorch export spike and PyTorch parity coverage.

## Evaluation policy

- Training fits model parameters and derives class weights.
- Development selects checkpoints and development-time decisions.
- Test is evaluated once after model and decision policy are frozen.
- Morphology reports overall and annotated exact accuracy.
- Per-label reports include support, precision, recall, F1, and Average
  Precision.
- Average Precision is calculated over the complete split, never averaged from
  mini-batch AP values.
- `<NONE>` labels are excluded from real-label macro summaries.
- A distilled student must be compared with the same student trained without
  teacher knowledge.

## Historical format-2 benchmarks

All checkpoints and measurements in the following benchmark sections use
checkpoint format 2 and the former uniformly binary morphology objective.
They remain the comparison baseline for the new architecture, but they cannot
be loaded into the format-3 hybrid morphology heads and must not be presented
as measurements of that implementation.

## Bokmål gold-only student

The control student uses gold UD targets only and no teacher. It trains
NorBERT4-xsmall jointly with UPOS, 18 morphology feature heads, and 622 lemma
edit rules.

### Unweighted five-epoch control

- checkpoint: `runs/nb-student-baseline/best.pt`
- selected epoch: 5
- development joint loss: 0.208129
- development UPOS accuracy: 98.51%
- development lemma-rule accuracy: 96.20%
- non-`<NONE>` morphology micro precision/recall/F1:
  93.53% / 88.87% / 91.14%
- non-`<NONE>` morphology macro F1: 77.99%
- non-`<NONE>` morphology macro Average Precision: 86.42%

### Selected class-weighted control

The controlled ablation uses the square root of the training-only
negative-to-positive ratio for morphology positive examples. Weights are
capped at the predeclared value 10.0. Development loss remains unweighted and
comparable.

- checkpoint: `runs/nb-student-weighted/best.pt`
- selected epoch: 5
- checkpoint size: 68,386,651 bytes
- development joint loss: 0.209187
- development UPOS accuracy: 98.49%
- development lemma-rule accuracy: 96.16%
- non-`<NONE>` morphology micro precision/recall/F1:
  87.32% / 96.12% / 91.51%
- non-`<NONE>` morphology macro F1: 89.64%
- non-`<NONE>` morphology macro Average Precision: 93.01%
- Average Precision improves for 37 of 40 real morphology labels

The threshold-independent gain confirms that weighting improves label ranking
rather than merely activating more values. This checkpoint is the current
gold-only student reference for Nynorsk and later teacher-distillation
comparisons. Final output thresholds and confidence calibration remain
deferred.

Complete feature and label results are recorded in `docs/benchmarks.md`.

## Nynorsk gold-only student

The Nynorsk-only parameter reference uses the same class-weighting policy and
the shared Norwegian schema derived exclusively from the Bokmål and Nynorsk
training splits. It receives only Nynorsk sentences during optimization.

- checkpoint: `runs/nn-student-weighted/best.pt`
- checkpoint format: 2
- schema language tags: `nb`, `nn`
- selected epoch: 5
- checkpoint size: 68,739,291 bytes
- development joint loss: 0.239937
- development UPOS accuracy: 98.13%
- development lemma-rule accuracy: 96.19%
- supported non-`<NONE>` morphology micro precision/recall/F1:
  82.27% / 93.96% / 87.73%
- supported non-`<NONE>` morphology macro F1: 81.94%
- supported non-`<NONE>` morphology macro Average Precision: 87.17%

The Nynorsk training split contains no positive `Gender=Com` example, while
the development split contains 733. The reference therefore reaches 0% F1
and 2.32% Average Precision for that value. Bokmål training contains 4,806
positive examples, making this the clearest predeclared test of whether shared
training transfers useful information across the written standards. Four of
the 40 real shared-schema labels have no Nynorsk development support and are
excluded from supported-label macro summaries. Both Norwegian test splits
remain untouched.

## Selected shared Norwegian gold-only student

The shared model optimizes one NorBERT4-xsmall student over the concatenated,
nearly balanced Bokmål and Nynorsk training splits. It uses the shared schema
and the same capped class-weighting policy as the single-standard controls.
Checkpoint selection uses the combined development loss, while quality is
reported separately for each written standard.

- checkpoint: `runs/no-student-weighted/best.pt`
- checkpoint format: 2
- model language tag: `no`
- schema language tags: `nb`, `nn`
- training sentences: 29,870
- combined development sentences: 4,299
- selected epoch: 5
- checkpoint size: 68,739,419 bytes
- end-to-end wall time: approximately 18 minutes 28 seconds

Bokmål development:

- joint loss: 0.176108
- UPOS accuracy: 98.64%
- lemma-rule accuracy: 96.70%
- morphology micro precision/recall/F1: 89.01% / 96.72% / 92.70%
- morphology macro F1: 91.67%
- morphology macro Average Precision: 94.07%

Nynorsk development:

- joint loss: 0.216549
- UPOS accuracy: 98.30%
- lemma-rule accuracy: 96.64%
- supported-label morphology micro precision/recall/F1:
  83.75% / 95.13% / 89.08%
- supported-label morphology macro F1: 86.18%
- supported-label morphology macro Average Precision: 90.44%

The shared student improves UPOS and lemma-rule accuracy over both
single-standard weighted controls. On Nynorsk, `Gender=Com` improves from 0%
F1 and 2.32% Average Precision to 9.77% F1 and 58.44% Average Precision. This
confirms useful cross-standard transfer without a measured Bokmål regression.
Final thresholds and confidence calibration remain deferred, and both test
splits remain untouched.

## Selected shared Norwegian teacher

The first quality-oriented teacher fine-tunes the Apache-2.0-licensed
`ltg/norbert4-base` backbone on the same joint gold data, schema, task heads,
and capped morphology weighting policy as the shared student. The backbone is
pinned at commit `386ba2dc5ae5f95fec86d580c5fc4af34d380126`.

- checkpoint: `runs/no-teacher-base/best.pt`
- model role: `teacher`
- parameters: 148,899,624 before Prism task heads
- hidden size: 640
- selected epoch: 4
- checkpoint size: 598,665,563 bytes
- end-to-end wall time: approximately 2 hours 20 minutes 31 seconds

Bokmål development:

- joint loss: 0.077641
- UPOS accuracy: 99.20%
- lemma-rule accuracy: 98.97%
- morphology micro precision/recall/F1: 96.17% / 98.65% / 97.40%
- morphology macro F1: 96.84%
- morphology macro Average Precision: 98.74%

Nynorsk development:

- joint loss: 0.118087
- UPOS accuracy: 98.90%
- lemma-rule accuracy: 98.87%
- supported-label morphology micro precision/recall/F1:
  92.66% / 97.13% / 94.84%
- supported-label morphology macro F1: 90.58%
- supported-label morphology macro Average Precision: 93.42%

The teacher improves both written standards over the selected shared student
and is accepted as the first distillation source. `Gender=Com` on Nynorsk has
62.67% Average Precision; its low fixed-threshold F1 remains a later decoding
and calibration concern. NorBERT4-large is deferred unless base distillation
fails to improve the compact student.

## Selected shared Norwegian distilled student

The first on-the-fly logit-distillation prototype is complete. It freezes the
selected NorBERT4-Base teacher and trains a fresh NorBERT4-xsmall student on
the same shared gold corpus. Teacher logits supplement, but do not replace,
the supervised UPOS, morphology, and lemma targets.

The first two policies were rejected:

- temperature 2.0, weight 0.5: the weighted teacher loss contributed about
  58% of the final training objective and reduced lemma quality and morphology
  recall;
- temperature 2.0, weight 0.1: the teacher contribution fell to about 26%,
  but both written standards still remained just below gold-only.

The selected first distilled reference uses:

- checkpoint: `runs/no-student-distilled-w010-t100/best.pt`
- teacher: `runs/no-teacher-base/best.pt`
- temperature: 1.0
- distillation weight: 0.1
- selected epoch: 5
- checkpoint size: 68,740,059 bytes
- end-to-end wall time: approximately 32 minutes 10 seconds

Bokmål development:

- joint loss: 0.175509
- UPOS accuracy: 98.66%
- lemma-rule accuracy: 96.71%
- morphology micro precision/recall/F1: 89.26% / 96.61% / 92.79%
- morphology macro F1: 91.81%
- morphology macro Average Precision: 94.04%

Nynorsk development:

- joint loss: 0.215845
- UPOS accuracy: 98.29%
- lemma-rule accuracy: 96.65%
- supported-label morphology micro precision/recall/F1:
  83.65% / 95.03% / 88.98%
- supported-label morphology macro F1: 86.19%
- supported-label morphology macro Average Precision: 90.41%

This is the first controlled evidence that the teacher can improve the compact
student. The gain is small and primarily shifts morphology toward higher
precision with slightly lower recall. A single temperature is not an adequate
long-term policy for 17-way UPOS, binary morphology decisions, and the
1,059-way lemma-rule head. Both official test splits remain untouched.

The controlled class-balanced morphology-distillation ablation is complete:

- checkpoint: `runs/no-student-distilled-balanced-w010-t100/best.pt`
- temperature: 1.0
- distillation weight: 0.1
- selected epoch: 5
- end-to-end wall time: approximately 34 minutes 32 seconds

Bokmål development:

- joint loss: 0.175663
- UPOS accuracy: 98.65%
- lemma-rule accuracy: 96.70%
- morphology micro precision/recall/F1: 88.98% / 96.73% / 92.70%
- morphology macro F1: 91.67%
- morphology macro Average Precision: 94.07%

Nynorsk development:

- joint loss: 0.216049
- UPOS accuracy: 98.29%
- lemma-rule accuracy: 96.65%
- supported-label morphology micro precision/recall/F1:
  83.34% / 95.16% / 88.86%
- supported-label morphology macro F1: 86.17%
- supported-label morphology macro Average Precision: 90.45%

This ablation is rejected as the selected reference. It raises recall and
ranking quality slightly for some rare labels, but loses precision and micro
F1 and does not produce a consistent UPOS or lemma improvement. The simpler
temperature-1.0, weight-0.1 distilled checkpoint remains the historical
format-2 distilled reference.

## Selected format-3 hybrid gold-only student

Checkpoint format 3 replaces the former uniformly binary morphology
formulation with a schema-driven hybrid contract while retaining the compact
linear task heads:

- 12 exclusive Norwegian features emit `<NONE>` plus their real values and
  train with categorical Cross-Entropy;
- 6 genuinely multi-valued features (`Case`, `Definite`, `Gender`, `Number`,
  `PronType`, and `VerbForm`) emit only real-value logits and train with Binary
  Cross-Entropy;
- `<NONE>` for a multi-valued feature is derived when no real value is active;
- supervised class weights and teacher distillation follow the same
  categorical-versus-binary split;
- evaluation converts both variants back into the complete public label space
  before accuracy, precision, recall, F1, and Average Precision are computed.

The implementation is covered by focused schema, head, decoding, loss,
weighting, distillation, training-step, epoch, and checkpoint-contract tests.
The first controlled joint Bokmål-Nynorsk run is accepted as the new gold-only
student reference:

- checkpoint: `runs/no-student-hybrid-weighted/best.pt`
- checkpoint format: 3
- selected epoch: 5
- checkpoint size: 68,735,067 bytes
- end-to-end wall time: approximately 20 minutes 47 seconds
- combined development loss: 0.192474

Bokmål development:

- joint loss: 0.173502 under the format-3 objective;
- UPOS accuracy: 98.64%;
- lemma-rule accuracy: 96.69%;
- morphology micro precision/recall/F1: 89.62% / 97.02% / 93.18%;
- morphology macro F1: 91.79%;
- morphology macro Average Precision: 95.25%.

Nynorsk development:

- joint loss: 0.214554 under the format-3 objective;
- UPOS accuracy: 98.36%;
- lemma-rule accuracy: 96.63%;
- supported-label morphology micro precision/recall/F1:
  84.32% / 95.62% / 89.61%;
- supported-label morphology macro F1: 86.74%;
- supported-label morphology macro Average Precision: 91.37%.

Relative to the selected format-2 gold-only reference, morphology improves on
both written standards. Bokmål gains 0.48 percentage points micro F1 and 1.18
points macro Average Precision. Nynorsk gains 0.53 points micro F1, 0.56
points macro F1, and 0.93 points macro Average Precision. UPOS and lemma remain
effectively stable. The format-2 and format-3 losses are not directly
comparable because their morphology objectives differ.

Format 2 is intentionally rejected by current checkpoint loaders because the
morphology head tensor shapes and their meanings changed. The existing
format-2 teacher and distilled checkpoints remain historical references and
cannot be used for format-3 distillation.

## Implemented token-pooling ablation

The selected format-3 reference uses the first contextualized subword vector
for every original token. A controlled Mean-pooling alternative is now
implemented without changing the backbone, task heads, loss policy, or output
schema:

- tokenized batches carry the start and exclusive end of every token's
  contiguous subword span;
- `first` gathers the span's first contextualized vector;
- `mean` averages all contextualized vectors in the span through a vectorized
  prefix-sum operation;
- `--token-pooling {first,mean}` selects the training policy;
- checkpoints store `token_pooling_strategy`, and evaluation restores it
  automatically;
- existing format-3 checkpoints without this metadata resolve explicitly to
  `first` and remain compatible.

Mean pooling adds no trainable parameters. It is implemented and tested but
has not yet been trained or benchmarked. Export parity for the complete tagger
remains required before a production runtime accepts either pooling path.

## Repeatable commands

Train the unweighted Bokmål control:

```bash
python -m prism.languages.norwegian.train_baseline
```

Train the selected shared format-3 gold-only student:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-hybrid-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Train the controlled Mean-pooling candidate:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --token-pooling mean \
  --checkpoint runs/no-student-hybrid-mean-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Evaluate the selected checkpoint on Bokmål development:

```bash
python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nb \
  --checkpoint runs/no-student-hybrid-weighted/best.pt \
  --analysis runs/no-student-hybrid-weighted/nb-development-analysis.json
```

## Repository cleanup decision

Historical recurrent, dictionary, and related command/test paths were removed
after the Transformer student surpassed their task scope and became the
required gold-only distillation reference. They are not part of the active
architecture and must not be reintroduced. Benchmark documentation now starts
with the Transformer student generation.

## Immediate next step

Train the controlled Mean-pooling student and evaluate it separately on
Bokmål and Nynorsk development. Compare it with the selected First-pooling
format-3 reference while holding every other training decision fixed. Only
after selecting the token-pooling policy should Prism test a nonlinear head or
train the expensive format-3 teacher. Do not evaluate either official test
split until the architecture and calibration policy are fixed.
