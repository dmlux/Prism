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
- a checkpoint-compatible `last` versus `learned-last-four` backbone-layer
  aggregation contract; the selected learned strategy adds four mixture logits
  and one scale parameter;
- shared normalization before task heads;
- UPOS, per-feature morphology, and lemma edit-rule heads;
- the selected
  `wide-shared-mlp-structured-morphology-character-cnn` architecture with a
  generic `H -> 2H -> H` residual projection, a parallel soft-decision
  morphology refinement, and a compact character path for morphology and
  lemma; the narrower variants remain checkpoint-compatible;
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
The first controlled joint Bokmål-Nynorsk run became the format-3
First-pooling reference:

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

## Selected Mean-pooling student

The first format-3 reference used the first contextualized subword vector for
every original token. A controlled Mean-pooling alternative was trained
without changing the backbone, task heads, loss policy, optimizer, seed, data,
or output schema:

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

Mean pooling adds no trainable parameters. The controlled run is accepted as
the new gold-only student reference:

- checkpoint: `runs/no-student-hybrid-mean-weighted/best.pt`
- checkpoint format: 3
- token pooling: `mean`
- selected epoch: 5
- checkpoint size: 68,735,131 bytes
- end-to-end wall time: approximately 21 minutes 8 seconds
- combined development loss: 0.189321

| Development metric | First, Bokmål | Mean, Bokmål | First, Nynorsk | Mean, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.173502 | **0.169886** | 0.214554 | **0.211940** |
| UPOS accuracy | 98.64% | **98.67%** | **98.36%** | 98.35% |
| Lemma-rule accuracy | 96.69% | **96.83%** | 96.63% | **96.70%** |
| Morphology micro precision | 89.62% | **89.83%** | 84.32% | **84.45%** |
| Morphology micro recall | 97.02% | **97.19%** | **95.62%** | 95.59% |
| Morphology micro F1 | 93.18% | **93.36%** | 89.61% | **89.67%** |
| Morphology macro F1 | 91.79% | **91.97%** | **86.74%** | 86.63% |
| Morphology macro Average Precision | 95.25% | **95.56%** | **91.37%** | 91.28% |

Mean pooling reduces the supervised loss and improves lemma accuracy,
morphology precision, and morphology micro F1 on both written standards. Its
small Nynorsk regressions in UPOS, recall, macro F1, and macro Average
Precision are all at most 0.11 percentage points. The broader two-standard
gain and unchanged model size justify selecting Mean pooling for new student
training. First pooling remains an explicit ablation option. Existing format-3
checkpoints without pooling metadata continue to resolve to `first`; they are
never silently reinterpreted. Export parity for the complete tagger remains
required before production runtime acceptance.

## Shared-MLP architecture family

The controlled Student candidate keeps the selected Mean pooling and every
existing task output contract. A single shared residual projection transforms
the normalized token vector before all linear task heads:

```text
head_input = normalized + dropout(gelu(linear(normalized)))
```

- `--task-head-architecture` selects `linear`, `shared-mlp`,
  `wide-shared-mlp`, `wide-shared-mlp-task-adapters`, or
  `wide-shared-mlp-structured-morphology`, or
  `wide-shared-mlp-structured-morphology-character-cnn`;
- `wide-shared-mlp-structured-morphology-character-cnn` is the default for new
  Norwegian training runs;
- `shared-mlp` adds 37,056 parameters for hidden size 192;
- `wide-shared-mlp` uses `192 -> 384 -> 192`, contains 148,032 projection
  parameters, and adds 110,976 parameters relative to `shared-mlp`;
- `wide-shared-mlp-task-adapters` preserves that shared projection and adds
  separate residual `192 -> 96 -> 192` adapters for UPOS, morphology, and
  lemma; the 18 morphology heads deliberately share one morphology adapter;
- the three adapters add 111,456 parameters, approximately 446 KB in FP32,
  and start as exact identity functions through zero-initialized output
  projections;
- `wide-shared-mlp-structured-morphology` preserves the selected shared
  projection without task adapters and adds a parallel soft-decision
  refinement pass over UPOS and every morphology feature;
- for the joint Norwegian schema the structured decoder reads 69 soft values
  and adds 23,476 parameters, approximately 94 KB in FP32;
- its feature correction heads start at zero, so the candidate initially
  reproduces the independent morphology logits exactly;
- checkpoints store `token_task_head_architecture`;
- evaluation and teacher loading restore the stored architecture;
- older format-3 checkpoints without the field resolve to `linear`;
- `--epoch-count` exposes the training duration with default 12.

The five-epoch candidate is accepted as the new gold-only Student reference:

- checkpoint: `runs/no-student-hybrid-mean-shared-mlp-weighted/best.pt`
- selected epoch: 5
- checkpoint size: 68,883,921 bytes
- end-to-end wall time: approximately 20 minutes 57 seconds
- combined development loss: 0.171392

| Development metric | Linear, Bokmål | Shared MLP, Bokmål | Linear, Nynorsk | Shared MLP, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.169886 | **0.152651** | 0.211940 | **0.193203** |
| UPOS accuracy | 98.67% | **98.71%** | 98.35% | **98.40%** |
| Lemma-rule accuracy | 96.83% | **97.13%** | 96.70% | **97.12%** |
| Morphology micro precision | 89.83% | **90.61%** | 84.45% | **85.18%** |
| Morphology micro recall | 97.19% | **97.23%** | 95.59% | **95.94%** |
| Morphology micro F1 | 93.36% | **93.80%** | 89.67% | **90.24%** |
| Morphology macro F1 | 91.97% | **92.63%** | 86.63% | **87.49%** |
| Morphology macro Average Precision | 95.56% | **96.06%** | 91.28% | **91.83%** |

The shared MLP improves every reported headline metric on both written
standards while adding only 148,790 checkpoint bytes and no measured training
time. The explicit `linear` option and the checkpoint fallback preserve the
controlled reference and compatibility.

## Selected eight-epoch training policy

The controlled duration ablation keeps the selected Mean pooling, shared MLP,
model initialization, data, optimizer, losses, seed, batch size, and
evaluation policy fixed. It changes the scheduled training duration from five
to eight epochs. The best checkpoint is again the final epoch, and every
headline metric improves on both written standards:

- checkpoint: `runs/no-student-hybrid-mean-shared-mlp-e8-weighted/best.pt`
- selected epoch: 8
- checkpoint size: 68,883,921 bytes
- end-to-end wall time: approximately 33 minutes 3 seconds
- combined development loss: 0.145512

| Development metric | 5 epochs, Bokmål | 8 epochs, Bokmål | 5 epochs, Nynorsk | 8 epochs, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.152651 | **0.124505** | 0.193203 | **0.169961** |
| UPOS accuracy | 98.71% | **98.86%** | 98.40% | **98.51%** |
| Lemma-rule accuracy | 97.13% | **97.73%** | 97.12% | **97.71%** |
| Morphology micro precision | 90.61% | **91.82%** | 85.18% | **86.75%** |
| Morphology micro recall | 97.23% | **97.59%** | 95.94% | **96.28%** |
| Morphology micro F1 | 93.80% | **94.62%** | 90.24% | **91.27%** |
| Morphology macro F1 | 92.63% | **93.62%** | 87.49% | **88.39%** |
| Morphology macro Average Precision | 96.06% | **97.05%** | 91.83% | **92.48%** |

Eight epochs are selected as the new default because the gains require no
additional model parameters or inference work. Development loss improved at
every epoch, including epoch 8, although the per-epoch gain was diminishing.
The subsequent ten-epoch run measures the remaining diminishing gain before
the final predeclared twelve-epoch boundary.

## Selected ten-epoch training policy

The next controlled duration run keeps every architecture, data, optimizer,
loss, seed, batching, and evaluation decision fixed while extending the
scheduled training duration from eight to ten epochs. Its final epoch is again
the best combined Development checkpoint:

- checkpoint: `runs/no-student-hybrid-mean-shared-mlp-e10-weighted/best.pt`
- selected epoch: 10
- checkpoint size: 68,883,921 bytes
- end-to-end wall time: approximately 40 minutes 59 seconds
- combined development loss: 0.138900

| Development metric | 8 epochs, Bokmål | 10 epochs, Bokmål | 8 epochs, Nynorsk | 10 epochs, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.124505 | **0.115954** | 0.169961 | **0.165604** |
| UPOS accuracy | 98.86% | **98.87%** | 98.51% | **98.54%** |
| Lemma-rule accuracy | 97.73% | **97.95%** | 97.71% | **97.87%** |
| Morphology micro precision | 91.82% | **92.41%** | 86.75% | **87.36%** |
| Morphology micro recall | 97.59% | **97.77%** | 96.28% | **96.41%** |
| Morphology micro F1 | 94.62% | **95.01%** | 91.27% | **91.66%** |
| Morphology macro F1 | 93.62% | **94.07%** | 88.39% | **88.67%** |
| Morphology macro Average Precision | 97.05% | **97.32%** | 92.48% | **92.58%** |

Ten epochs become the new default because every headline metric improves on
both written standards without increasing model size or inference work. The
gains are diminishing, so one twelve-epoch run is predeclared as the final
duration ablation. Prism will not continue increasing epoch count based on the
same Development splits after that result.

## Selected twelve-epoch training policy

The final predeclared duration ablation extends the unchanged selected
training policy from ten to twelve scheduled epochs. The final epoch again has
the lowest combined Development loss:

- checkpoint: `runs/no-student-hybrid-mean-shared-mlp-e12-weighted/best.pt`
- selected epoch: 12
- checkpoint size: 68,883,921 bytes
- end-to-end wall time: approximately 49 minutes
- combined development loss: 0.134762

| Development metric | 10 epochs, Bokmål | 12 epochs, Bokmål | 10 epochs, Nynorsk | 12 epochs, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.115954 | **0.110285** | 0.165604 | **0.163249** |
| UPOS accuracy | 98.87% | **98.93%** | **98.54%** | 98.53% |
| Lemma-rule accuracy | 97.95% | **98.16%** | 97.87% | **98.03%** |
| Morphology micro precision | 92.41% | **92.87%** | 87.36% | **88.04%** |
| Morphology micro recall | 97.77% | **97.86%** | 96.41% | **96.50%** |
| Morphology micro F1 | 95.01% | **95.30%** | 91.66% | **92.08%** |
| Morphology macro F1 | 94.07% | **94.55%** | 88.67% | **88.98%** |
| Morphology macro Average Precision | 97.32% | **97.53%** | 92.58% | **92.71%** |

Twelve epochs are selected as the final duration policy. Nynorsk UPOS falls by
only 0.0128 percentage points while Loss, Lemma, and all morphology summaries
improve on both standards. As predeclared, epoch-count tuning on these
Development splits is now closed.

## Selected wide shared-MLP student

The controlled capacity ablation keeps Mean pooling, the twelve-epoch
schedule, backbone, data, seed, optimizer, losses, output heads, checkpoint
selection, and evaluation policy fixed. It replaces only the shared
`192 -> 192` projection with `192 -> 384 -> 192`.

- checkpoint: `runs/no-student-hybrid-mean-wide-shared-mlp-e12-weighted/best.pt`
- selected checkpoint: scheduled epoch 10 of 12
- checkpoint size: 69,328,391 bytes
- end-to-end wall time: approximately 49 minutes 16 seconds
- combined development loss: 0.133798

| Development metric | Narrow, Bokmål | Wide, Bokmål | Narrow, Nynorsk | Wide, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.110285 | **0.103405** | **0.163249** | 0.169170 |
| UPOS accuracy | **98.93%** | 98.92% | 98.53% | 98.53% |
| Lemma-rule accuracy | 98.16% | **98.38%** | 98.03% | **98.10%** |
| Morphology micro precision | 92.87% | **93.36%** | 88.04% | **88.82%** |
| Morphology micro recall | 97.86% | **98.15%** | 96.50% | **96.59%** |
| Morphology micro F1 | 95.30% | **95.70%** | 92.08% | **92.54%** |
| Morphology macro F1 | 94.55% | **94.78%** | **88.98%** | 88.92% |
| Morphology macro Average Precision | 97.53% | **97.86%** | 92.71% | **92.93%** |

The wider projection is selected because it improves morphology micro F1,
Average Precision, and lemma accuracy on both written standards for only
444,470 additional checkpoint bytes. Bokmål UPOS and Nynorsk macro F1 regress
by only 0.0055 and 0.0618 percentage points, while Nynorsk UPOS is unchanged.
The higher Nynorsk Loss affects all task components despite better discrete
predictions; this is recorded as a raw-confidence and calibration risk rather
than hidden by the selection.

## Selected learned final-four layer mixture

The controlled aggregation ablation keeps Mean pooling, the twelve-epoch
schedule, wide shared MLP, backbone, data, seed, optimizer, losses, checkpoint
selection, and evaluation policy fixed. It replaces the final backbone layer
with a learned scalar mixture of the final four layers.

- checkpoint: `runs/no-student-hybrid-last4-mean-wide-shared-mlp-e12-weighted/best.pt`
- selected checkpoint: scheduled epoch 12 of 12
- checkpoint size: 69,329,021 bytes, only 630 bytes above the control
- combined development loss: 0.132484 instead of 0.133798
- learned layer weights from `-4` through `-1`: 21.05%, 16.31%, 23.38%,
  39.25%
- learned scale: 1.8578

| Development metric | Final layer, Bokmål | Learned mix, Bokmål | Final layer, Nynorsk | Learned mix, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.103405 | **0.102312** | 0.169170 | **0.167598** |
| UPOS accuracy | 98.92% | **98.98%** | 98.53% | **98.65%** |
| Lemma-rule accuracy | 98.38% | **98.45%** | 98.10% | **98.22%** |
| Morphology micro precision | 93.36% | **93.66%** | 88.59% | **88.89%** |
| Morphology micro recall | **98.15%** | 98.09% | **96.59%** | 96.57% |
| Morphology micro F1 | 95.70% | **95.83%** | 92.41% | **92.57%** |
| Morphology macro F1 | 94.78% | **95.11%** | 88.92% | **89.54%** |
| Morphology macro Average Precision | 97.86% | **97.99%** | **92.93%** | 92.61% |

The learned mixture is selected as the default for new Norwegian training
runs. Existing checkpoints without aggregation metadata remain `last`; their
meaning does not change. Small recall regressions and the Nynorsk macro Average
Precision regression remain recorded calibration risks.

## Rejected task-family adapter ablation

The `wide-shared-mlp-task-adapters` candidate adds three residual
`192 -> 96 -> 192` task-family paths after the selected shared projection. Its
best checkpoint is epoch 7 of 12, has a combined Development loss of 0.131067,
and is 69,778,113 bytes. This is 449,092 bytes larger than the selected
control.

| Development metric | Control, Bokmål | Adapters, Bokmål | Control, Nynorsk | Adapters, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | **0.102312** | 0.109405 | 0.167598 | **0.156277** |
| UPOS accuracy | **98.98%** | 98.87% | **98.65%** | 98.61% |
| Lemma-rule accuracy | **98.45%** | 98.09% | **98.22%** | 98.07% |
| Morphology micro precision | **93.66%** | 93.29% | 88.89% | **88.99%** |
| Morphology micro recall | 98.09% | **98.16%** | 96.57% | **96.72%** |
| Morphology micro F1 | **95.83%** | 95.66% | 92.57% | **92.69%** |
| Morphology macro F1 | **95.11%** | 94.57% | **89.54%** | 89.19% |
| Morphology macro Average Precision | 97.99% | **98.05%** | 92.61% | **93.29%** |

The candidate is rejected because the Nynorsk morphology gains do not offset
regressions in Nynorsk UPOS, lemma, and macro F1 or the broader Bokmål losses.
The adapter implementation remains available only as a reproducible ablation
and does not become the default.

## Selected structured morphology decoder

The `wide-shared-mlp-structured-morphology` ablation keeps the selected
learned-last-four aggregation, Mean pooling, wide shared MLP, twelve-epoch
schedule, data, seed, optimizer, losses, and output contracts fixed. Its
parallel second pass conditions residual morphology corrections on soft UPOS
and morphology distributions without a hard decision cascade.

- checkpoint:
  `runs/no-student-hybrid-last4-mean-wide-shared-mlp-structured-morphology-e12-weighted/best.pt`
- selected checkpoint: scheduled epoch 12 of 12
- checkpoint size: 69,434,687 bytes, 105,666 bytes above the control
- combined Development loss: 0.130524 instead of 0.132484
- end-to-end wall time: approximately 53 minutes 20 seconds

| Development metric | Control, Bokmål | Structured, Bokmål | Control, Nynorsk | Structured, Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.102312 | **0.100505** | 0.167598 | **0.165460** |
| UPOS accuracy | **98.98%** | **98.98%** | **98.65%** | 98.62% |
| Lemma-rule accuracy | 98.45% | **98.48%** | 98.22% | **98.31%** |
| Morphology micro precision | 93.66% | **93.73%** | 88.89% | **89.29%** |
| Morphology micro recall | 98.09% | **98.34%** | 96.57% | **96.80%** |
| Morphology micro F1 | 95.83% | **95.98%** | 92.57% | **92.89%** |
| Morphology macro F1 | 95.11% | **95.28%** | **89.54%** | 89.50% |
| Morphology macro Average Precision | 97.99% | **98.11%** | 92.61% | **93.03%** |

The structured decoder is selected as the new gold-only student architecture.
It improves Loss, Lemma, morphology precision, recall, micro F1, and Average
Precision on both written standards and improves Bokmål macro F1. Bokmål UPOS
is unchanged. The Nynorsk UPOS and macro-F1 regressions are only 0.0256 and
0.0470 percentage points, while the primary Nynorsk morphology micro F1 gains
0.3254 points. The 105,666-byte size increase is approximately 0.15%. The
complete structured tagger passes strict `torch.export` capture with output
parity. Both official test splits remain untouched.

The preceding independent-head control passes strict `torch.export`, XNNPACK
lowering to an 86,641,292-byte ExecuTorch `.pte`, and portable-runtime
execution. The
graph includes backbone, layer mixture, Mean pooling, wide shared MLP, and all
20 logit outputs. Against PyTorch, the maximum absolute output error is
`1.91e-5`; the largest mean absolute error among its outputs is `7.95e-6`.
Dynamic shapes, production backend parity, peak memory, and document-scale
performance are still open release gates.

## Repeatable commands

Train an unweighted Bokmål model with the current defaults:

```bash
python -m prism.languages.norwegian.train_baseline
```

Train the selected shared format-3 gold-only student:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --backbone-layer-aggregation learned-last-four \
  --token-pooling mean \
  --task-head-architecture wide-shared-mlp-structured-morphology \
  --epoch-count 12 \
  --checkpoint runs/no-student-hybrid-last4-mean-wide-shared-mlp-structured-morphology-e12-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Reproduce the rejected task-adapter candidate:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --backbone-layer-aggregation learned-last-four \
  --token-pooling mean \
  --task-head-architecture wide-shared-mlp-task-adapters \
  --epoch-count 12 \
  --checkpoint runs/no-student-hybrid-last4-mean-wide-shared-mlp-task-adapters-e12-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Reproduce the structured-morphology control explicitly:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --backbone-layer-aggregation learned-last-four \
  --token-pooling mean \
  --task-head-architecture wide-shared-mlp-structured-morphology \
  --epoch-count 12 \
  --checkpoint runs/no-student-hybrid-last4-mean-wide-shared-mlp-structured-morphology-e12-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Reproduce the rejected First-pooling control explicitly:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --token-pooling first \
  --checkpoint runs/no-student-hybrid-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Evaluate the selected checkpoint on Bokmål development:

```bash
python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nb \
  --checkpoint runs/no-student-hybrid-last4-mean-wide-shared-mlp-structured-morphology-e12-weighted/best.pt \
  --analysis runs/no-student-hybrid-last4-mean-wide-shared-mlp-structured-morphology-e12-weighted/nb-development-analysis.json
```

Reproduce the rejected linear-head control explicitly:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --token-pooling mean \
  --task-head-architecture linear \
  --epoch-count 5 \
  --checkpoint runs/no-student-hybrid-mean-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Reproduce the ten-epoch duration control explicitly:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --token-pooling mean \
  --task-head-architecture shared-mlp \
  --epoch-count 10 \
  --checkpoint runs/no-student-hybrid-mean-shared-mlp-e10-weighted/best.pt \
  --morphology-weight-cap 10.0
```

## Repository cleanup decision

Historical recurrent, dictionary, and related command/test paths were removed
after the Transformer student surpassed their task scope and became the
required gold-only distillation reference. They are not part of the active
architecture and must not be reintroduced. Benchmark documentation now starts
with the Transformer student generation.

## Immediate next step

The language-independent Rare/OOV evaluation contract is implemented. It
normalizes complete token forms with Unicode NFC plus case folding, defines
`rare` as one to five occurrences in the checkpoint's complete schema-training
corpus, and defines `oov` as zero occurrences. Evaluation keeps every full
sentence as model context and restricts only metric accumulation. It reports
UPOS, morphology micro precision/recall/F1, representable lemma-rule accuracy,
lemma-rule coverage, and end-to-end lemma accuracy for both slices.

The Bokmål development control is recorded: 3,150 rare tokens achieve 98.3810%
UPOS, 94.2222% end-to-end lemma accuracy, and 91.9243% morphology micro F1;
2,807 OOV tokens achieve 97.9694% UPOS, 91.5212% end-to-end lemma accuracy, and
91.4391% morphology micro F1. The largest clear gap is OOV lemmatization rather
than UPOS, so the compact character branch should primarily enrich lemma and
morphology while preserving the shared NorBERT4 sentence representation.

The Nynorsk control is also recorded: 2,393 rare tokens achieve 98.0359% UPOS,
94.4421% end-to-end lemma accuracy, and 86.7812% morphology micro F1; 2,536 OOV
tokens achieve 97.2397% UPOS, 90.6940% end-to-end lemma accuracy, and 84.9270%
morphology micro F1. Nynorsk therefore strengthens the case for feeding the
character representation into both lemma and morphology.

The compact character-aware branch is implemented. Its language-independent,
versioned vocabulary normalizes tokens with Unicode NFC, represents unknown
characters and word boundaries explicitly, and preserves prefix and suffix
around a truncation marker. A 32-dimensional embedding feeds parallel width-3
and width-5 convolutions whose masked maximum produces a 192-dimensional token
vector. A residual fusion feeds only morphology and lemma; UPOS remains on the
unchanged contextual path. For the joint Norwegian training corpus the 120
literal characters produce a 125-ID vocabulary. Encoder and fusion add 102,688
parameters (410,752 raw FP32 bytes). Checkpoints store the vocabulary and
32-position token limit, existing checkpoints remain compatible, and the flat
character-aware model contract passes strict `torch.export` parity.

The twelve-epoch gold-only character-CNN training run completed in about 54
minutes 55 seconds. Development-loss selection chose epoch 8 at a combined
loss of 0.112245, improving on the selected structured control's 0.130524.
The 69,862,812-byte checkpoint is 428,125 bytes larger than the control and
remains below the 100 MB target.

Bokmål development confirms the targeted effect. Overall loss falls from
0.100505 to 0.086190 and lemma-rule accuracy rises from 98.4811% to 98.8223%,
while UPOS changes slightly from 98.9771% to 98.9469%. Rare end-to-end lemma
accuracy rises by 2.6667 percentage points and morphology micro F1 by 1.7586
points. OOV end-to-end lemma accuracy rises by 1.3894 points, morphology micro
F1 by 1.2246 points, and UPOS by 0.2493 points. Rare UPOS decreases by only
0.0635 points.

Nynorsk development independently confirms the targeted effect. Loss falls
from 0.165460 to 0.142569. Rare end-to-end lemma accuracy rises by 2.4238
percentage points and morphology micro F1 by 1.5048 points. OOV UPOS rises by
0.1183 points, OOV lemma end-to-end by 0.1183 points, and OOV morphology micro
F1 by 0.8214 points. Overall lemma-rule accuracy gains 0.2274 points; overall
UPOS trades 0.0576 points.

The character-aware branch is therefore selected as the new gold-only
standard and the default for new Norwegian training. It meets its declared
Rare/OOV objective on both written standards, lowers both development losses,
and keeps the tiny overall-UPOS tradeoffs explicit. Neither official test
split was evaluated.

## Accepted architecture ablation plan

Preserve the selected learned aggregation and evaluate exactly these stages in
order:

1. small task-family-specific residual adapters for UPOS, morphology, and
   lemma: completed and rejected;
2. a structured morphology decoder that models dependencies between feature
   decisions without a hard UPOS error cascade: completed and selected;
3. a compact character-aware branch for rare and previously unseen word forms,
   feeding lemma and morphology while preserving the shared NorBERT4 token
   representation: completed and selected.

Each stage received its own checkpoint and separate Bokmål/Nynorsk report.
The character-aware ablation also reports the implemented Rare/OOV slices;
aggregate gains alone were not used as evidence of its intended benefit.

The character-aware format-3 Teacher completed twelve epochs. Development-loss
selection chose epoch 3 at 0.098218; its 609,180,828-byte checkpoint is stored
at `runs/no-teacher-base-character-cnn-e12-weighted/best.pt`. The run took
approximately 3 hours 29 minutes 36 seconds.
Confidence calibration remains a separate final stage, especially
because the selected wide MLP improves discrete Nynorsk quality while
worsening raw negative log-likelihood.

Bokmål Teacher evaluation is complete. Against the selected character-aware
Student, overall UPOS improves by 0.3355 percentage points and lemma-rule
accuracy by 0.1513 points. Rare morphology micro F1 improves by 2.5205 points;
OOV UPOS, lemma end-to-end, and morphology micro F1 improve by 0.3563, 0.8550,
and 2.1551 points respectively.

Nynorsk also passes the acceptance gate. Overall UPOS and lemma-rule accuracy
improve by 0.2496 and 0.1249 points. Rare morphology micro F1 improves by
3.0982 points; OOV UPOS, lemma end-to-end, and morphology micro F1 improve by
0.5915, 1.6956, and 4.0556 points. The Teacher is accepted for format-3
distillation because it beats the fixed gold-only Student on every reported
aggregate and Rare/OOV comparison metric on both written standards.

The first character-aware format-3 distillation run is complete. It uses
temperature 1.0 and weight 0.1, selected epoch 8 of 12, and wrote the
69,863,132-byte checkpoint
`runs/no-student-character-cnn-distilled-w010-t100-e12-weighted/best.pt`. Its
joint development loss of 0.109941 is lower than the fixed gold-only control's
0.112245. This is promising but not a selection result until separate
Bokmål/Nynorsk and Rare/OOV evaluation is complete. The CLI defaults now match
the historically selected 1.0/0.1 policy instead of the rejected 2.0/0.5
starting point.

Bokmål evaluation of the distilled Student is complete and mixed. Development
loss improves from 0.086190 to 0.084777 and overall lemma-rule accuracy by
0.0192 percentage points; overall UPOS, OOV UPOS, and OOV lemma end-to-end are
unchanged. Rare lemma gains 0.1905 points and OOV morphology micro F1 gains
0.0744 points, while Rare UPOS and morphology micro F1 regress by 0.0953 and
0.0411 points. This result alone was insufficient for selection; the following
Nynorsk evaluation completed the decision.

Nynorsk improves on every reported comparison metric. Development loss falls
from 0.142569 to 0.139228; overall UPOS and lemma-rule accuracy gain 0.0256 and
0.0448 percentage points. Rare UPOS, lemma end-to-end, and morphology micro F1
gain 0.0418, 0.0417, and 0.2073 points. OOV UPOS, lemma end-to-end, and
morphology micro F1 gain 0.1183, 0.3549, and 0.0308 points.

The distilled Student is selected as the new compact reference. Both written
standards improve in loss and lemma, no OOV metric regresses, Bokmål overall
UPOS remains unchanged, and Nynorsk UPOS improves. The small Bokmål Rare-UPOS
and Rare-morphology regressions remain explicit. The Teacher-to-Student gap is
still large, so any later distillation refinement must use task-specific
policies rather than another blind global sweep.

The first task-specific distillation-policy implementation is complete and
benchmarked. `TokenTaskDistillationPolicy` independently controls the
temperature and loss weight for UPOS, morphology, and lemma while retaining
the selected global 1.0/0.1 CLI values as backward-compatible fallbacks.
Training steps and epochs consume the typed policy, new checkpoints serialize
all six resolved values, and the CLI prints them before a distilled run. This
does not change the Student architecture, checkpoint parameter count, export,
or inference cost. That first implementation deliberately preceded full DKD
and did not yet separate target-class and non-target-class losses; the later
DKD implementation below now adds that separation.

The controlled candidate kept all temperatures at 1.0, lowered UPOS
distillation to 0.05, raised morphology distillation to 0.20, and retained
lemma distillation at 0.10. Epoch 8 again produced the best checkpoint at
`runs/no-student-character-cnn-distilled-task-policy-e12-weighted/best.pt`;
joint development loss improved by only 0.000064 to 0.109877.

Separate evaluation rejects the candidate. On Bokmål, loss improves by
0.000652 and Rare UPOS by 0.1270 percentage points, but overall UPOS and lemma,
Rare morphology, OOV lemma, and OOV morphology regress. The OOV-lemma drop is
0.2494 points. On Nynorsk, Rare/OOV morphology micro F1 improves by 0.3127 and
0.0549 points and Rare lemma by 0.0418 points, but loss worsens by 0.000620;
overall UPOS and lemma, Rare UPOS, and OOV lemma also regress. The candidate
therefore fails the two-standard acceptance rule. The uniform temperature-1.0,
weight-0.1 Student remains the selected compact reference. The official test
splits remain untouched.

True categorical DKD is now implemented as an optional, language-independent
distillation objective. It separates target-class knowledge (TCKD) from the
renormalized non-target distribution (NCKD), exposes independent component
weights, and keeps the selected classic KL objective as the default. UPOS,
lemma rules, and exclusive morphology features use DKD when selected;
multi-value morphology remains on binary KL because it has no single target
class. Gold target IDs flow through the typed training step, the resolved
objective and component weights are checkpointed, and focused loss, masking,
gradient, CLI, and end-to-end training-step tests cover the contract.

The first DKD ablation is complete and selected as the new compact Student
reference. It restored the uniform outer policy (temperature 1.0 and task
weight 0.1) and used TCKD/NCKD component weights of 1.0/1.0. Epoch 12 selected
`runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/best.pt` at
69,863,388 bytes. Joint development loss falls from the uniform-KL
reference's 0.109941 to 0.101139.

On Bokmål, loss falls from 0.084777 to 0.077811, overall UPOS gains 0.0165
percentage points, and lemma gains 0.1816 points. Rare/OOV morphology micro F1
gains 0.8449/0.5952 points and Rare/OOV lemma end-to-end gains 0.4444/0.1781
points. OOV UPOS regresses by 0.1781 points.

On Nynorsk, loss falls from 0.139228 to 0.128288, overall UPOS gains 0.0448
points, and lemma gains 0.0545 points. Rare/OOV morphology micro F1 gains
0.9569/0.9343 points and Rare/OOV lemma end-to-end gains 0.1672/0.4337 points.
Rare and OOV UPOS regress by 0.1254 and 0.0788 points. The much broader
two-standard gains select DKD while keeping these localized UPOS tradeoffs
explicit. The model architecture and inference cost are unchanged, and both
official test splits remain untouched.
