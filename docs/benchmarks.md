# Prism Benchmarks

Unless a section explicitly identifies checkpoint format 3, the measurements
below are historical format-2 references using the former uniformly binary
morphology objective. They remain the fixed comparison for the hybrid
morphology architecture implemented on 2026-07-21.

## Norwegian Bokmål Transformer student

All results use the official gold tokenization and the original
train, development, and test splits.

### Dataset

- Source: UniversalDependencies/UD_Norwegian-Bokmaal
- Commit: `396d11f0c2bd290a2a2711015c04ac25bc3dcc06`
- License: CC BY-SA 4.0
- Training sentences: 15,696
- Development sentences: 2,409
- Test sentences: 1,939

The test split remains reserved for final evaluation after architecture,
training policy, confidence calibration, and artifact configuration are fixed.

### NorBERT4-xsmall student without distillation

This is the first end-to-end Transformer student baseline. It was trained only
from the gold UD targets; no teacher or distillation loss was used. It is the
required control model for a later comparison against the same student trained
with teacher knowledge. The test split was not evaluated.

### Training configuration

- Backbone: `ltg/norbert4-xsmall`, pinned through the Bokmål profile
- Python: 3.12.13
- PyTorch: 2.12.1
- Device: Apple MPS on arm64, macOS 26.5.2
- Epochs: 1
- Training batch size: 16
- Training sentences/tokens: 15,696 / 243,886
- Development sentences/tokens: 2,409 / 36,369
- Optimizer: differential AdamW
- Backbone learning rate: 0.00002
- Task-head learning rate: 0.0005
- Weight decay: 0.01
- Gradient clipping: 1.0
- Linear warmup ratio: 0.1
- Random seed: 42
- End-to-end command wall time: approximately 2 minutes 7 seconds
- Checkpoint size: 68,427,851 bytes (approximately 65.3 MiB)

### Loss and headline development metrics

| Metric | Result |
| --- | ---: |
| Training joint loss | 1.497864 |
| Development joint loss | 0.535158 |
| Development UPOS accuracy | 96.89% |
| Development lemma-rule accuracy | 91.03% |

### Development morphology accuracy

Overall accuracy includes the frequent `<NONE>` target. Annotated accuracy is
therefore the important companion result and shows that one epoch is not yet a
competitive morphology model.

| Feature | Overall | Annotated |
| --- | ---: | ---: |
| Abbr | 99.56% | 0.00% |
| Animacy | 99.60% | 90.89% |
| Case | 96.89% | 38.65% |
| Definite | 84.08% | 41.35% |
| Degree | 97.08% | 72.01% |
| Foreign | 99.86% | 0.00% |
| Gender | 73.24% | 16.55% |
| Mood | 99.27% | 95.07% |
| NumType | 99.04% | 40.34% |
| Number | 68.71% | 23.44% |
| Person | 97.95% | 69.34% |
| Polarity | 99.66% | 58.11% |
| Poss | 99.30% | 0.00% |
| PronType | 93.30% | 44.10% |
| Reflex | 99.61% | 0.00% |
| Tense | 98.98% | 91.59% |
| VerbForm | 87.49% | 26.32% |
| Voice | 99.62% | 0.00% |

The 0% annotated results occur on sparse features for which the one-epoch
student still predicts `<NONE>` rather than an annotated value. Further epochs,
per-value metrics, and class-imbalance analysis are required before selecting
training policy or comparing against a teacher.

### Five-epoch selected baseline

The repeatable multi-epoch command trained five epochs from a fresh initialized
task-head state, evaluated the development split after every epoch, and
atomically retained the checkpoint with the lowest joint development loss. The
development loss improved in every epoch (`0.448856`, `0.289774`, `0.239129`,
`0.214759`, `0.208129`), so epoch 5 was selected. The test split remained
unused.

- End-to-end wall time: approximately 15 minutes 42 seconds
- Selected checkpoint: epoch 5
- Checkpoint size: 68,384,923 bytes (approximately 65.2 MiB)
- Training joint loss at selected epoch: 0.230259
- Development joint loss: 0.208129
- Development UPOS accuracy: 98.51%
- Development lemma-rule accuracy: 96.20%

| Feature | Overall | Annotated |
| --- | ---: | ---: |
| Abbr | 99.58% | 11.25% |
| Animacy | 99.94% | 98.72% |
| Case | 99.78% | 96.62% |
| Definite | 97.48% | 93.31% |
| Degree | 99.03% | 94.30% |
| Foreign | 99.93% | 60.00% |
| Gender | 87.97% | 62.44% |
| Mood | 99.62% | 97.87% |
| NumType | 99.78% | 88.62% |
| Number | 96.28% | 92.88% |
| Person | 99.78% | 97.20% |
| Polarity | 99.99% | 99.32% |
| Poss | 99.96% | 95.65% |
| PronType | 98.96% | 91.86% |
| Reflex | 100.00% | 100.00% |
| Tense | 99.58% | 97.33% |
| VerbForm | 98.72% | 93.53% |
| Voice | 99.75% | 37.68% |

UPOS and lemma improve strongly over the one-epoch smoke baseline. The table
uses the corrected multi-value decoder: each non-`<NONE>` output is activated
at a logit of zero, which is the natural 0.5 probability boundary for the
binary-cross-entropy objective. The earlier decoder compared every value logit
with the `<NONE>` logit. That caused widespread false-positive values whenever
both logits were negative and understated exact annotated accuracy for all six
multi-valued features. The corrected decoder raises annotated accuracy for
Case from 68.44% to 96.62%, Definite from 64.36% to 93.31%, Gender from 10.37%
to 62.44%, Number from 30.19% to 92.88%, PronType from 9.90% to 91.86%, and
VerbForm from 26.55% to 93.53%, without retraining or changing loss values.

The remaining weaknesses are now much more specific: rare labels still have
low recall, including `Abbr=Yes`, `Mood=Imp`, `NumType=Ord`, `Voice=Pass`, and
several rare PronType values. `Gender=Fem` also has only 4.88% recall despite
substantial development support. These per-value failures, rather than a
general `<NONE>` or multi-value failure, are the next supervised-baseline
quality target before teacher distillation.

### Five-epoch class-weighted ablation

A fresh student used the same seed, data, architecture, optimizer, schedule,
and five-epoch policy as the selected unweighted baseline. Only morphology
positive examples were reweighted by the square root of the training-split
negative-to-positive ratio. Weights were fixed from training data alone and
capped at 10.0 before the run; no development or test labels selected the cap.
Development loss remained unweighted and comparable. The test split remained
unused.

- End-to-end wall time: approximately 9 minutes 34 seconds
- Selected checkpoint: epoch 5
- Checkpoint: `runs/nb-student-weighted/best.pt`
- Checkpoint size: 68,386,651 bytes (approximately 65.2 MiB)
- Weighted training joint loss at selected epoch: 0.239275
- Unweighted development joint loss: 0.209187
- Development UPOS accuracy: 98.49%
- Development lemma-rule accuracy: 96.16%

| Feature | Overall | Annotated |
| --- | ---: | ---: |
| Abbr | 99.62% | 70.63% |
| Animacy | 99.96% | 99.40% |
| Case | 99.71% | 99.11% |
| Definite | 97.77% | 96.76% |
| Degree | 99.04% | 96.53% |
| Foreign | 99.93% | 84.00% |
| Gender | 85.63% | 58.33% |
| Mood | 99.68% | 98.52% |
| NumType | 99.90% | 98.45% |
| Number | 96.19% | 93.53% |
| Person | 99.87% | 99.19% |
| Polarity | 99.99% | 100.00% |
| Poss | 99.97% | 98.42% |
| PronType | 99.09% | 94.60% |
| Reflex | 100.00% | 100.00% |
| Tense | 99.68% | 98.59% |
| VerbForm | 98.44% | 94.88% |
| Voice | 99.93% | 94.93% |

Across all non-`<NONE>` morphology labels, micro precision changes from 93.53%
to 87.32%, recall from 88.87% to 96.12%, and micro F1 from 91.14% to 91.51%.
Macro F1 rises substantially from 77.99% to 89.64%, showing that rare labels
benefit rather than merely increasing prediction volume. In particular,
`Abbr=Yes` F1 rises from 19.15% to 62.09%, `Mood=Imp` from 0% to 47.62%,
`NumType=Ord` from 48.82% to 91.49%, `PronType=Rcp` from 0% to 92.31%, and
`Voice=Pass` from 53.33% to 90.97%.

Gender exact annotated accuracy falls from 62.44% to 58.33%, but every real
Gender label improves in F1: Com 68.02% to 68.42%, Fem 9.19% to 44.82%, Masc
76.55% to 81.18%, and Neut 78.23% to 83.28%. The exact-match regression is
therefore caused by additional simultaneous value activations, especially the
low-precision `Gender=Fem` output, not by worse per-label recognition. The
weighted checkpoint is a strong rare-label candidate, but its precision-recall
tradeoff must remain explicit when selecting a production model.

The threshold-independent ranking result confirms that the gain is learned
rather than merely caused by activating more labels. Macro Average Precision
over all 40 real, non-`<NONE>` morphology labels increases from 86.42% to
93.01%. The weighted student improves Average Precision for 37 labels, leaves
`Reflex=Yes` perfect, changes `Person=1` only by numerical noise, and decreases
only `PronType=Prs` materially, from 99.62% to 99.39%. Strong rare-label AP
gains include `Mood=Imp` from 11.24% to 46.97%, `NumType=Ord` from 69.38% to
97.06%, `PronType=Rcp` from 0.59% to 85.74%, and `Voice=Pass` from 79.98% to
88.63%. This makes the capped class-weighted checkpoint the stronger
supervised student baseline for subsequent Nynorsk and teacher-distillation
comparisons, while final output calibration remains deferred.

## Norwegian Nynorsk Transformer student

### Five-epoch class-weighted single-standard reference

This reference optimizes model parameters only on the official Nynorsk
training split. Its task heads use the shared Norwegian schema built from the
pinned Bokmål and Nynorsk training splits, so later separate and joint models
have directly compatible output inventories. No development labels contribute
to the schema or class weights, and the test split remains unused.

- Source: UniversalDependencies/UD_Norwegian-Nynorsk
- Commit: `aaeb9d90c748c2bd9e272f180b599484f9f05ac6`
- License: CC BY-SA 4.0
- Training sentences/tokens: 14,174 / 245,330
- Development sentences/tokens: 1,890 / 31,250
- Shared-schema sentences: 29,870
- Shared morphology features: 18
- Shared lemma edit rules: 1,059
- Epochs: 5
- Batch size: 16
- Morphology positive-weight cap: 10.0
- End-to-end wall time: approximately 9 minutes 1 second
- Selected checkpoint: epoch 5
- Checkpoint: `runs/nn-student-weighted/best.pt`
- Checkpoint size: 68,739,291 bytes
- Weighted training joint loss at selected epoch: 0.264932
- Unweighted development joint loss: 0.239937
- Development UPOS accuracy: 98.13%
- Development lemma-rule accuracy: 96.19%
- Supported real morphology labels: 36 of 40
- Supported-label micro precision/recall/F1: 82.27% / 93.96% / 87.73%
- Supported-label macro F1: 81.94%
- Supported-label macro Average Precision: 87.17%

The main controlled weakness is `Gender=Com`: it has 733 development examples
but no positive Nynorsk training example, producing 0% F1 and 2.32% Average
Precision. The Bokmål training split contains 4,806 positive examples, so this
label is a direct transfer test for the upcoming balanced joint model. Other
low-AP supported labels include `Number=Sing` at 0.02% with only two examples,
`Gender=Fem` at 46.68%, `Foreign=Yes` at 53.90%, and `Mood=Imp` at 53.97%.
Labels without Nynorsk development support have undefined Average Precision
and are excluded from the supported-label macro summaries.

## Shared Norwegian Transformer student

### Five-epoch class-weighted joint reference

This reference trains one NorBERT4-xsmall student on the concatenated Bokmål
and Nynorsk training splits. The two corpora contain similar token counts, and
the shared schema and class weights are derived exclusively from their
training data. Checkpoint selection uses the combined development loss;
reported quality remains separate for each written standard.

- Training sentences: 29,870
- Combined development sentences: 4,299
- Shared morphology features: 18
- Shared lemma edit rules: 1,059
- Epochs: 5
- Batch size: 16
- Morphology positive-weight cap: 10.0
- End-to-end wall time: approximately 18 minutes 28 seconds
- Selected checkpoint: epoch 5
- Checkpoint: `runs/no-student-weighted/best.pt`
- Checkpoint size: 68,739,419 bytes

| Development metric | Bokmål | Nynorsk |
| --- | ---: | ---: |
| Joint loss | 0.176108 | 0.216549 |
| UPOS accuracy | 98.64% | 98.30% |
| Lemma-rule accuracy | 96.70% | 96.64% |
| Morphology micro precision | 89.01% | 83.75% |
| Morphology micro recall | 96.72% | 95.13% |
| Morphology micro F1 | 92.70% | 89.08% |
| Morphology macro F1 | 91.67% | 86.18% |
| Morphology macro Average Precision | 94.07% | 90.44% |

Nynorsk macro values exclude the four real labels without development
support. All 40 shared-schema labels have Bokmål development support. The
joint student improves UPOS and lemma-rule accuracy over both corresponding
single-standard controls. `Gender=Com` on Nynorsk improves from 0% to 9.77%
F1 and from 2.32% to 58.44% Average Precision, demonstrating useful transfer
from Bokmål training data. Both official test splits remain untouched.

### First-pooling format-3 hybrid morphology reference

This controlled successor keeps the same shared NorBERT4-xsmall backbone,
training data, seed, optimizer, five-epoch schedule, linear task heads, and
class-weight cap. It changes only the morphology output contract: exclusive
features use categorical softmax/Cross-Entropy, while genuinely multi-valued
features use sigmoid/Binary Cross-Entropy over real values and derive
`<NONE>`.

- Checkpoint format: 3
- Selected checkpoint: epoch 5
- Checkpoint: `runs/no-student-hybrid-weighted/best.pt`
- Checkpoint size: 68,735,067 bytes
- End-to-end wall time: approximately 20 minutes 47 seconds
- Combined development loss: 0.192474 under the format-3 objective

| Development metric | Bokmål | Nynorsk |
| --- | ---: | ---: |
| Joint loss | 0.173502 | 0.214554 |
| UPOS accuracy | 98.64% | 98.36% |
| Lemma-rule accuracy | 96.69% | 96.63% |
| Morphology micro precision | 89.62% | 84.32% |
| Morphology micro recall | 97.02% | 95.62% |
| Morphology micro F1 | 93.18% | 89.61% |
| Morphology macro F1 | 91.79% | 86.74% |
| Morphology macro Average Precision | 95.25% | 91.37% |

Nynorsk summaries exclude the four real labels without development support.
The hybrid contract improves morphology precision, recall, micro F1, macro F1,
and macro Average Precision on both written standards while preserving UPOS
and lemma quality. It became the controlled First-pooling reference for the
subsequent token-pooling ablation. Joint loss
must not be compared numerically across formats because the morphology loss
changed. Both official test splits remain untouched.

### Selected Mean-pooling format-3 student

This controlled ablation changes only subword-to-token pooling. Instead of
using the first contextualized subword state, it averages every contextualized
state in the original token's contiguous subword span. Backbone, schema,
linear task heads, objectives, optimizer, seed, data, and five-epoch schedule
remain unchanged.

- Selected checkpoint: epoch 5
- Checkpoint: `runs/no-student-hybrid-mean-weighted/best.pt`
- Checkpoint size: 68,735,131 bytes
- End-to-end wall time: approximately 21 minutes 8 seconds
- Combined development loss: 0.189321

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

Mean pooling is selected as the new gold-only student standard because it
reduces development loss and improves lemma accuracy, morphology precision,
and morphology micro F1 on both written standards without adding parameters
or material training time. Nynorsk UPOS, recall, macro F1, and macro Average
Precision regress by at most 0.11 percentage points; these small tradeoffs are
recorded rather than hidden. Both official test splits remain untouched.

### Selected shared-MLP format-3 student

This controlled ablation keeps Mean pooling and changes only the shared input
path to the task heads. A residual `Linear(192, 192) -> GELU -> Dropout` block
adds 37,056 parameters before the existing schema-driven linear heads.

- Selected checkpoint: epoch 5
- Checkpoint: `runs/no-student-hybrid-mean-shared-mlp-weighted/best.pt`
- Checkpoint size: 68,883,921 bytes
- End-to-end wall time: approximately 20 minutes 57 seconds
- Combined development loss: 0.171392

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

The shared MLP improves every headline metric on both written standards while
adding only 148,790 checkpoint bytes and no measured wall-time cost. It is
therefore selected as the new gold-only Student architecture. The next
controlled ablation changes only the training duration from five to eight
epochs. Both official test splits remain untouched.

### Selected eight-epoch shared-MLP student

This controlled duration ablation keeps Mean pooling, the shared MLP, model
initialization, data, optimizer, losses, seed, batch size, and evaluation
policy fixed. The configured linear warmup-decay schedule spans eight instead
of five epochs, and checkpoint selection still uses combined development loss.

- Selected checkpoint: epoch 8
- Checkpoint: `runs/no-student-hybrid-mean-shared-mlp-e8-weighted/best.pt`
- Checkpoint size: 68,883,921 bytes
- End-to-end wall time: approximately 33 minutes 3 seconds
- Combined development loss: 0.145512

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

Eight epochs improve every headline metric on both written standards without
changing checkpoint size or inference cost, so eight becomes the default
training duration. Development loss still improved at epoch 8, but by a
smaller amount than in prior epochs. One controlled ten-epoch run will test
the remaining gain before model capacity or teacher training changes. Both
official test splits remain untouched.

### Selected ten-epoch shared-MLP student

This duration ablation extends the otherwise unchanged selected architecture
and training policy from eight to ten scheduled epochs. Combined Development
loss selects the final epoch again.

- Selected checkpoint: epoch 10
- Checkpoint: `runs/no-student-hybrid-mean-shared-mlp-e10-weighted/best.pt`
- Checkpoint size: 68,883,921 bytes
- End-to-end wall time: approximately 40 minutes 59 seconds
- Combined development loss: 0.138900

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

Ten epochs improve every headline metric again without changing model size or
inference cost and therefore become the new default. The gains are now
diminishing. One predeclared twelve-epoch run is the final duration ablation;
epoch count will not be tuned further on these Development splits afterward.
Both official test splits remain untouched.

### Selected twelve-epoch shared-MLP student

The final predeclared duration ablation extends the unchanged ten-epoch policy
to twelve scheduled epochs. Combined Development loss again selects the final
epoch.

- Selected checkpoint: epoch 12
- Checkpoint: `runs/no-student-hybrid-mean-shared-mlp-e12-weighted/best.pt`
- Checkpoint size: 68,883,921 bytes
- End-to-end wall time: approximately 49 minutes
- Combined development loss: 0.134762

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

Twelve epochs become the final duration policy. Nynorsk UPOS trades 0.0128
percentage points for improvements in Loss, Lemma, and every morphology
summary on both standards. The predeclared stopping rule closes epoch-count
tuning on these Development splits. Both official test splits remain
untouched.

### Selected wider shared-MLP student

The controlled `wide-shared-mlp` candidate changes only the shared residual
projection from `192 -> 192` to `192 -> 384 -> 192`. It adds 148,032
projection parameters in total, approximately 592 KB in FP32. Relative to the
selected 37,056-parameter projection, the increase is 110,976 parameters, or
approximately 444 KB in FP32. The ablation preserves Mean pooling, the
twelve-epoch schedule, data, initialization seed, optimizer, losses, output
heads, checkpoint selection, and evaluation policy. Existing `shared-mlp`
checkpoints remain unchanged and are the control.

- Selected checkpoint: scheduled epoch 10 of 12
- Checkpoint: `runs/no-student-hybrid-mean-wide-shared-mlp-e12-weighted/best.pt`
- Checkpoint size: 69,328,391 bytes
- End-to-end wall time: approximately 49 minutes 16 seconds
- Combined development loss: 0.133798

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

The wider projection is selected. It improves the primary morphology micro F1
and threshold-independent Average Precision on both standards, improves lemma
accuracy on both, and costs only 444,470 additional checkpoint bytes. Nynorsk
UPOS is unchanged; Bokmål UPOS and Nynorsk macro F1 trade 0.0055 and 0.0618
percentage points respectively. Nynorsk Loss rises across all three task
components, indicating a raw-confidence tradeoff that must be measured during
the already required calibration stage. Both official test splits remain
untouched.

### Selected learned final-four layer mixture

This controlled ablation changes only the backbone output aggregation from the
final layer to a learned scalar mixture of the final four layers. Mean pooling,
the wide shared MLP, twelve epochs, data, seed, optimizer, losses, checkpoint
selection, and evaluation policy remain fixed.

- Selected checkpoint: scheduled epoch 12 of 12
- Checkpoint: `runs/no-student-hybrid-last4-mean-wide-shared-mlp-e12-weighted/best.pt`
- Checkpoint size: 69,329,021 bytes
- Combined development loss: 0.132484
- Learned weights from layers `-4` through `-1`: 21.05%, 16.31%, 23.38%,
  39.25%
- Learned output scale: 1.8578

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

The learned mixture is selected. It improves Loss, UPOS, Lemma, morphology
precision, micro F1, and macro F1 on both written standards for only 630
additional checkpoint bytes. The small recall regressions and the 0.33-point
Nynorsk macro Average Precision regression remain explicit tradeoffs. Most of
the Nynorsk ranking regression is concentrated in `Gender=Com`; its discrete
F1 still improves from 7.48% to 8.18%, but this label remains unreliable.

The trained complete tagger passes strict `torch.export` with exact
exported-program parity for the measured input. Backbone, layer mixture, Mean
pooling, wide shared MLP, and all 20 logit outputs lower to an 86,641,292-byte
XNNPACK ExecuTorch `.pte`. It executes through the portable runtime with a
maximum absolute output error of `1.91e-5`; the largest mean absolute error
among its outputs is `7.95e-6`. This fixed-shape spike confirms exportability,
but does not yet accept dynamic input shapes, production backend parity, peak
memory, or document-scale performance.

### Rejected task-family adapter candidate

The controlled `wide-shared-mlp-task-adapters` ablation keeps the selected
learned final-four mixture, Mean pooling, wide shared MLP, twelve-epoch
schedule, data, seed, optimizer, losses, checkpoint selection, and evaluation
policy fixed. It adds separate residual `192 -> 96 -> 192` adapters for UPOS,
morphology, and lemma. The 18 morphology heads share one adapter.

- Selected checkpoint: scheduled epoch 7 of 12
- Checkpoint:
  `runs/no-student-hybrid-last4-mean-wide-shared-mlp-task-adapters-e12-weighted/best.pt`
- Checkpoint size: 69,778,113 bytes, 449,092 bytes above the control
- Combined Development loss: 0.131067 instead of 0.132484

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

The candidate is rejected. Its lower combined loss hides an uneven tradeoff:
Nynorsk loss, morphology micro F1, and ranking quality improve, but Nynorsk
UPOS, lemma accuracy, and macro F1 regress. Bokmål loses loss, UPOS, lemma,
precision, micro F1, and macro F1 for only 0.06-point recall and Average
Precision gains. The extra 449 KB therefore do not produce a robust
cross-standard improvement. The implementation remains an explicit
reproducible ablation option; `wide-shared-mlp` remains the independent-head
control for the next ablation. Both official test splits remain untouched.

### Selected structured morphology decoder

The controlled `wide-shared-mlp-structured-morphology` ablation keeps the
selected learned final-four mixture, Mean pooling, wide shared MLP,
twelve-epoch schedule, data, seed, optimizer, losses, checkpoint selection,
and evaluation policy fixed. It adds a parallel residual refinement pass that
reads soft UPOS and morphology distributions and predicts corrections for all
morphology features without hard decisions or an autoregressive order.

- Selected checkpoint: scheduled epoch 12 of 12
- Checkpoint:
  `runs/no-student-hybrid-last4-mean-wide-shared-mlp-structured-morphology-e12-weighted/best.pt`
- Checkpoint size: 69,434,687 bytes, 105,666 bytes above the control
- End-to-end wall time: approximately 53 minutes 20 seconds
- Combined Development loss: 0.130524 instead of 0.132484

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

The structured decoder is selected. It improves Loss, Lemma, morphology
precision, recall, micro F1, and Average Precision on both standards, while
Bokmål UPOS is unchanged. Nynorsk UPOS and macro F1 trade 0.0256 and 0.0470
percentage points respectively. The primary Nynorsk morphology micro F1 gains
0.3254 points, and the checkpoint grows by only approximately 0.15%. The next
controlled architecture ablation therefore uses this decoder as its control.
Both official test splits remain untouched.

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

### Rare/OOV evaluation contract

The selected structured checkpoint is also the fixed control for the planned
character-aware branch. Its dedicated development slices use only complete
token-form frequencies from the checkpoint's joint schema-training corpus:

- forms are normalized with Unicode NFC and case folding;
- `rare` means one to five normalized training occurrences;
- `oov` means zero normalized training occurrences and does not mean an
  unknown tokenizer subword;
- the model still receives each complete sentence; the slice mask restricts
  metric accumulation only;
- Bokmål and Nynorsk are reported separately without touching either test
  split.

Each slice reports token count, official-compatible `UPOS`, exact
complete-bundle `UFeats`, and `Lemmas` F1, plus UPOS accuracy, morphology micro
precision/recall/F1 over non-`<NONE>` labels, representable lemma-rule
accuracy, lemma-rule coverage, and end-to-end lemma accuracy. The final metric
counts an annotated lemma with a rule absent from the training schema as an
error instead of silently dropping it. The official slice metrics use the
same lemma restoration and explicit UD morphology-output policy as the
complete split.

The aligned external feature report is intentionally optional. Supplying
`--morphology-feature-comparison PATH` reads an already generated CoNLL-U
prediction and emits the full Prism-versus-system tables. Omitting the option
skips that work and runs only the faster standalone Prism evaluation; this
command never launches UDPipe itself.

The Bokmål and Nynorsk development controls were recorded from the selected
epoch-12 checkpoint without retraining:

| Standard | Token-frequency slice | Tokens | UPOS accuracy | Lemma-rule coverage | Lemma-rule accuracy | Lemma end-to-end accuracy | Morphology micro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bokmål | Rare | 3,150 | 98.3810% | 99.9048% | 94.3120% | 94.2222% | 91.9243% |
| Bokmål | OOV | 2,807 | 97.9694% | 99.4656% | 92.0129% | 91.5212% | 91.4391% |
| Nynorsk | Rare | 2,393 | 98.0359% | 99.8328% | 94.6002% | 94.4421% | 86.7812% |
| Nynorsk | OOV | 2,536 | 97.2397% | 99.2508% | 91.3786% | 90.6940% | 84.9270% |

The OOV result identifies lemmatization as the clearest weakness: compared
with the overall representable lemma-rule accuracy of 98.4811%, OOV
lemma-rule accuracy is lower by 6.47 percentage points. OOV UPOS remains much
more robust at 97.9694%, only 1.01 percentage points below the overall UPOS
accuracy. This supports feeding the planned character representation primarily
into lemma and morphology while retaining NorBERT4 sentence context. Nynorsk
shows the same lemma gap and a substantially larger OOV morphology gap: its
OOV morphology micro F1 is 84.9270%, while OOV UPOS remains 97.2397%.

### Selected character-CNN branch

The export-friendly character branch is the selected gold-only architecture. It
adds a training-derived, NFC-normalized 125-ID vocabulary for the joint
Norwegian corpus, 32-dimensional character embeddings, parallel width-3 and
width-5 convolutions, masked max pooling, and a residual fusion into morphology
and lemma only. UPOS remains on the unchanged contextual path. The branch adds
102,688 parameters (410,752 raw FP32 bytes) and passes strict `torch.export`
output parity.

The controlled run keeps data, seed, learned-last-four aggregation, Mean
pooling, structured morphology, loss policy, and twelve-epoch schedule fixed:

```bash
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --backbone-layer-aggregation learned-last-four \
  --token-pooling mean \
  --task-head-architecture wide-shared-mlp-structured-morphology-character-cnn \
  --epoch-count 12 \
  --checkpoint runs/no-student-character-cnn-e12-weighted/best.pt \
  --morphology-weight-cap 10.0
```

Selection requires separate Bokmål and Nynorsk development results plus the
fixed Rare/OOV slices above. The official test splits remain untouched.

The controlled training run completed in approximately 54 minutes 55 seconds.
Development-loss selection chose scheduled epoch 8 of 12 at a combined loss
of 0.112245, compared with 0.130524 for the selected structured control. The
checkpoint contains 69,862,812 bytes, an increase of 428,125 bytes over the
control. These promising joint-training values do not select the branch by
themselves.

The Bokmål development evaluation shows that the intended Rare/OOV gains are
real:

| Bokmål metric | Structured control | Character CNN | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.100505 | **0.086190** | -0.014315 |
| Overall UPOS accuracy | **98.9771%** | 98.9469% | -0.0302 pp |
| Overall lemma-rule accuracy | 98.4811% | **98.8223%** | +0.3412 pp |
| Rare UPOS accuracy | **98.3810%** | 98.3175% | -0.0635 pp |
| Rare lemma end-to-end accuracy | 94.2222% | **96.8889%** | +2.6667 pp |
| Rare morphology micro F1 | 91.9243% | **93.6829%** | +1.7586 pp |
| OOV UPOS accuracy | 97.9694% | **98.2187%** | +0.2493 pp |
| OOV lemma end-to-end accuracy | 91.5212% | **92.9106%** | +1.3894 pp |
| OOV morphology micro F1 | 91.4391% | **92.6637%** | +1.2246 pp |

The tiny Rare-UPOS regression is outweighed on Bokmål by large targeted lemma
and morphology gains, lower overall loss, and improved OOV UPOS.

Nynorsk independently confirms the intended effect:

| Nynorsk metric | Structured control | Character CNN | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.165460 | **0.142569** | -0.022891 |
| Overall UPOS accuracy | **98.6240%** | 98.5664% | -0.0576 pp |
| Overall lemma-rule accuracy | 98.3090% | **98.5364%** | +0.2274 pp |
| Rare UPOS accuracy | 98.0359% | **98.2031%** | +0.1672 pp |
| Rare lemma end-to-end accuracy | 94.4421% | **96.8659%** | +2.4238 pp |
| Rare morphology micro F1 | 86.7812% | **88.2860%** | +1.5048 pp |
| OOV UPOS accuracy | 97.2397% | **97.3580%** | +0.1183 pp |
| OOV lemma end-to-end accuracy | 90.6940% | **90.8123%** | +0.1183 pp |
| OOV morphology micro F1 | 84.9270% | **85.7484%** | +0.8214 pp |

The branch is selected because both written standards improve on the
predeclared Rare/OOV lemma and morphology targets, both lower development
loss, and OOV UPOS also improves. The small overall-UPOS tradeoffs of 0.0302
percentage points on Bokmål and 0.0576 points on Nynorsk remain explicit.

## Character-aware format-3 NorBERT4-Base teacher

The teacher matching the selected gold-only student architecture completed its
twelve-epoch joint Bokmål/Nynorsk training run. It uses Mean pooling, learned
last-four aggregation, the wide shared MLP, structured morphology, and the
selected character CNN. Development-loss selection chose epoch 3:

- Checkpoint: `runs/no-teacher-base-character-cnn-e12-weighted/best.pt`
- Selected checkpoint: epoch 3 of 12
- Joint development loss: 0.098218
- Joint UPOS accuracy at selection: 99.0668%
- Joint lemma-rule accuracy at selection: 98.8293%
- Checkpoint size: 609,180,828 bytes
- End-to-end wall time: approximately 3 hours 29 minutes 36 seconds

Later epochs improved some discrete metrics but had higher development loss;
the predeclared loss-based checkpoint policy was retained. Separate Bokmål and
Nynorsk development evaluations, including Rare/OOV slices, are required
before accepting this teacher for the new distillation ablation. Both official
test splits remain untouched.

The Bokmål development evaluation confirms that the Teacher is stronger than
the selected character-aware xsmall Student:

| Bokmål metric | Gold-only Student | Format-3 Teacher | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.086190 | **0.073450** | -0.012740 |
| Overall UPOS accuracy | 98.9469% | **99.2824%** | +0.3355 pp |
| Overall lemma-rule accuracy | 98.8223% | **98.9736%** | +0.1513 pp |
| Rare UPOS accuracy | 98.3175% | **99.0159%** | +0.6984 pp |
| Rare lemma end-to-end accuracy | 96.8889% | **96.9206%** | +0.0317 pp |
| Rare morphology micro F1 | 93.6829% | **96.2034%** | +2.5205 pp |
| OOV UPOS accuracy | 98.2187% | **98.5750%** | +0.3563 pp |
| OOV lemma end-to-end accuracy | 92.9106% | **93.7656%** | +0.8550 pp |
| OOV morphology micro F1 | 92.6637% | **94.8188%** | +2.1551 pp |

This is a useful Teacher signal on Bokmål, including the intended Rare/OOV
slices. The following Nynorsk evaluation completes the acceptance decision.

Nynorsk also shows a consistent Teacher advantage:

| Nynorsk metric | Gold-only Student | Format-3 Teacher | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.142569 | **0.127044** | -0.015525 |
| Overall UPOS accuracy | 98.5664% | **98.8160%** | +0.2496 pp |
| Overall lemma-rule accuracy | 98.5364% | **98.6613%** | +0.1249 pp |
| Rare UPOS accuracy | 98.2031% | **98.8299%** | +0.6268 pp |
| Rare lemma end-to-end accuracy | 96.8659% | **97.2837%** | +0.4178 pp |
| Rare morphology micro F1 | 88.2860% | **91.3842%** | +3.0982 pp |
| OOV UPOS accuracy | 97.3580% | **97.9495%** | +0.5915 pp |
| OOV lemma end-to-end accuracy | 90.8123% | **92.5079%** | +1.6956 pp |
| OOV morphology micro F1 | 85.7484% | **89.8040%** | +4.0556 pp |

The Teacher is therefore accepted for the format-3 distillation ablation. It
beats the same gold-only Student on every reported aggregate and Rare/OOV
comparison metric across both written standards. This establishes useful
Teacher headroom; it does not yet prove that the compact distilled Student
will retain the gain.

The later official-compatible Bokmål evaluation exposes an important
silver-labeling constraint. With canonical output and no morphology-logit
correction, the Teacher reaches 99.2824% UPOS F1, 95.3532% complete-bundle
UFeats F1, and 98.9057% Lemmas F1. Compared with reproduced UDPipe 2.17, this
is +0.3327 percentage points on UPOS, -2.7166 points on UFeats, and -0.0687
points on Lemmas. The Teacher therefore does not beat UDPipe on every task.
Its raw UFeats is also below the later selected Bundle-32 Student's corrected
96.0076%, because this Teacher predates the bundle reranker and the selected
logit-correction policy. Raw Teacher morphology must not be promoted to silver
gold without a controlled correction evaluation or a final architecture-
matched Teacher.

Applying the already selected full class-weight correction raises Teacher
Bokmål UFeats from 95.3532% to 96.3953%, a gain of 1.0421 percentage points.
It also raises Rare morphology micro F1 from 96.2034% to 96.5756% and OOV
morphology micro F1 from 94.8188% to 95.6278%. UPOS and Lemmas remain
unchanged by construction. The corrected Teacher is 0.3877 points above the
selected Bundle-32 Student on Bokmål UFeats, but still trails UDPipe by 1.6745
points; its Lemmas remain 0.0687 points behind UDPipe. Correction therefore
makes the existing Teacher a substantially safer morphology labeler, not an
across-the-board oracle. The equivalent Nynorsk gate is still required before
choosing the silver-label source.

The corrected canonical Nynorsk gate reaches 98.8160% UPOS, 94.4896% UFeats,
and 98.5888% Lemmas. This is +0.1440, +0.8512, and +0.0544 percentage points
over the selected canonical Bundle-32 Student. Rare and OOV morphology micro
F1 reach 91.9859% and 90.3169%, beating that Student by 0.6123 and 1.8500
points. Against UDPipe, Teacher UPOS leads by 0.2432 points while UFeats and
Lemmas trail by 1.1712 and 0.2400 points. The existing corrected Teacher is
therefore useful on both written standards, but it still predates the selected
Bundle-32 component. Before spending substantially more compute on labeling
50 million silver tokens, Prism will train one final architecture-matched
Bundle-32 Teacher and require it to beat this fixed corrected control.

The architecture-matched Base Teacher stopped manually during epoch 6 after
epoch 3 remained the loss-selected checkpoint and epochs 4 and 5 both failed
to improve. This is equivalent to the later planned patience-2 early-stopping
decision. Its checkpoint is
`runs/no-teacher-base-character-cnn-bundle32-e12-weighted/best.pt`
(609,716,974 bytes).

With full correction, Bundle-32 changes the historical Teacher as follows:

| Canonical Development metric | Old Teacher Bokmål | Bundle-32 Bokmål | Change | Old Teacher Nynorsk | Bundle-32 Nynorsk | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development loss | **0.073450** | 0.076953 | +0.003503 | **0.127044** | 0.134581 | +0.007537 |
| UPOS F1 | **99.2824%** | 99.2356% | -0.0468 pp | 98.8160% | **98.9312%** | +0.1152 pp |
| UFeats F1 | 96.3953% | **96.6235%** | +0.2282 pp | 94.4896% | **94.7296%** | +0.2400 pp |
| Lemmas F1 | **98.9057%** | 98.8589% | -0.0468 pp | **98.5888%** | 98.5504% | -0.0384 pp |
| Rare morphology micro F1 | 96.5756% | **96.6680%** | +0.0924 pp | 91.9859% | **92.2271%** | +0.2412 pp |
| OOV morphology micro F1 | **95.6278%** | 95.5126% | -0.1152 pp | 90.3169% | **90.4918%** | +0.1749 pp |

The Bundle-32 Teacher is selected as the primary Base morphology control
because exact UFeats and Rare morphology improve on both written standards,
and Nynorsk OOV morphology also improves. It is not treated as universally
superior: the old corrected Teacher remains the Base agreement control,
especially for Bokmål UPOS, Lemmas, and OOV morphology. Silver labels will not
silently splice task outputs from the two models; any agreement rule must be a
typed, benchmarked confidence policy.

The architecture-matched NorBERT4-large run is deferred. The Base result still
trails UDPipe by 1.4463 percentage points on Bokmål UFeats and by 0.1155 points
on Bokmål Lemmas; increasing backbone capacity has not been established as the
limiting variable. More importantly, the current summed-loss checkpoint
selection chose epoch 3 even though lemma-rule accuracy continued to improve
through epoch 5, and the bundle reranker has no direct complete-bundle loss
matching official UFeats. The next comparison is therefore a task-aligned
lemma/ranking audit and then a controlled Base objective or decoder ablation,
not another multi-hour capacity run.

The next predeclared compact-model comparison adds direct complete-bundle
supervision to the selected Bundle-32 Student. It uses auxiliary weight `0.1`,
a maximum of 12 epochs, and early-stopping patience 4. The old loss is
reproduced with weight `0`. Checkpoint selection remains lowest combined
Development loss; no UDPipe score enters optimization or selection. Acceptance
requires a material canonical UFeats improvement on both Bokmål and Nynorsk
without material regression in UPOS, Lemmas, per-feature morphology, or
Rare/OOV quality. Results are intentionally absent until the fixed run and both
Development evaluations complete. A longer convergence run is justified only
if the new best checkpoint again occurs at epoch 12.

The training half is complete. The run took approximately 1 hour 50 minutes
35 seconds and selected epoch 12, writing the 70,068,398-byte checkpoint
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-w010-e12-weighted/best.pt`.
Bundle loss decreased from 0.222202 to 0.112506 while the covered-token ratio
remained 98.2180%. Combined Development loss is 0.112011, but it includes the
new auxiliary term and must not be compared numerically with the old loss.
Canonical Bokmål evaluation is complete:

| Canonical Bokmål metric | Bundle-32 control | Direct bundle loss | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | **99.0046%** | 98.9964% | -0.0082 pp |
| UFeats F1 | 96.0076% | **96.7390%** | +0.7314 pp |
| Lemmas F1 | **98.9607%** | 98.8974% | -0.0632 pp |
| Rare morphology micro F1 | 95.5196% | **96.2849%** | +0.7653 pp |
| OOV morphology micro F1 | 94.1716% | **94.8654%** | +0.6938 pp |
| Rare lemma end-to-end | **97.2381%** | 97.1111% | -0.1270 pp |
| OOV lemma end-to-end | **93.6231%** | 92.8393% | -0.7838 pp |

Against UDPipe 2.17, the new Student leads Bokmål UPOS by 0.0467 points,
trails UFeats by 1.3308 points instead of 2.0622, and trails Lemmas by 0.0770
points. The auxiliary objective therefore closes 0.7314 points, or 35.5%, of
the remaining canonical UFeats gap in one controlled change. Its OOV-lemma
regression prevents immediate selection or schedule extension. Nynorsk remains
the second half of the predeclared acceptance gate.

Canonical Nynorsk evaluation is also complete:

| Canonical Nynorsk metric | Bundle-32 control | Direct bundle loss | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | 98.6720% | **98.6752%** | +0.0032 pp |
| UFeats F1 | 93.6384% | **94.4672%** | +0.8288 pp |
| Lemmas F1 | **98.5344%** | 98.5056% | -0.0288 pp |
| Rare morphology micro F1 | 91.3736% | **92.3979%** | +1.0243 pp |
| OOV morphology micro F1 | 88.4669% | **89.2680%** | +0.8011 pp |
| Rare lemma end-to-end | **96.9494%** | 96.9076% | -0.0418 pp |
| OOV lemma end-to-end | **91.6404%** | 91.5615% | -0.0789 pp |

Against UDPipe 2.17, the new Student leads Nynorsk UPOS by 0.1024 points,
trails UFeats by 1.1936 points instead of 2.0224, and trails Lemmas by 0.3232
points. The objective closes 0.8288 points, or 41.0%, of the prior Nynorsk
UFeats gap.

The consistent UFeats and Rare/OOV morphology gains on both standards validate
the direct objective. The consistent lemma tradeoff, especially the
0.7838-point Bokmål OOV-lemma regression, rejects this checkpoint as the
all-task reference. The run will not be extended. The next matched ablation
keeps the objective but restricts its gradient to the bundle residual scorer,
protecting the shared representation and independent task heads. The official
test splits remain untouched.

The residual-scorer-only ablation is implemented and predeclared with exactly
one additional switch:

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-direct-loss-isolated-w010-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --categorical-distillation-objective dkd \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --morphology-bundle-candidate-count 32 \
  --morphology-bundle-loss-weight 0.1 \
  --isolate-morphology-bundle-loss-gradient \
  --early-stopping-patience 4 \
  --epoch-count 12 \
  --morphology-weight-cap 10.0
```

The auxiliary loss receives the same forward scores as the unisolated run,
while only the candidate residual projection receives its gradients. Results
remain intentionally absent until training and both canonical Development
evaluations complete.

The training half completed in approximately 1 hour 50 minutes 59 seconds and
again selected epoch 12. The checkpoint is 70,068,462 bytes. Joint Development
training signals are:

| Joint Development signal | Bundle-32 control | Direct loss | Isolated direct loss |
| --- | ---: | ---: | ---: |
| UPOS loss | 0.046729 | **0.046595** | 0.046849 |
| Morphology loss | 0.008672 | **0.008133** | 0.008676 |
| Lemma-rule loss | 0.045614 | 0.046031 | **0.045161** |
| Lemma-rule accuracy | 98.8426% | 98.7938% | **98.8618%** |
| Bundle loss | not active | **0.112506** | 0.132241 |

The higher isolated bundle loss is expected because only the residual scorer
can optimize it. The improved lemma signal indicates that the intended
gradient protection is active, but canonical per-standard metrics remain the
acceptance gate. Since the best checkpoint is again the final epoch, a longer
schedule is eligible only if Bokmål and Nynorsk first accept this formulation.

Canonical Bokmål evaluation is complete:

| Canonical Bokmål metric | Bundle-32 control | Direct loss | Isolated loss |
| --- | ---: | ---: | ---: |
| UPOS F1 | **99.0046%** | 98.9964% | 98.9854% |
| UFeats F1 | 96.0076% | **96.7390%** | 96.0351% |
| Lemmas F1 | 98.9607% | 98.8974% | **98.9881%** |
| Rare morphology micro F1 | 95.5196% | **96.2849%** | 95.6446% |
| OOV morphology micro F1 | 94.1716% | **94.8654%** | 94.2055% |
| Rare lemma end-to-end | 97.2381% | 97.1111% | **97.4603%** |
| OOV lemma end-to-end | 93.6231% | 92.8393% | **93.6587%** |

Against the selected control, isolation gains 0.0275 UFeats points, 0.0274
Lemmas points, 0.1250/0.0339 Rare/OOV morphology points, and 0.2222/0.0356
Rare/OOV lemma points. Overall UPOS loses 0.0192 points while both frequency
slices are unchanged. The candidate still leads UDPipe 2.17 UPOS by 0.0357
points and now leads its Lemmas by 0.0137 points, but trails UFeats by 2.0347
points.

Against the unisolated candidate, the isolated loss recovers 0.0907 Lemmas
points and 0.8194 OOV-lemma points, while retaining only 0.0275 of the original
0.7314-point UFeats gain over Bundle-32. This validates the gradient-conflict
diagnosis and suggests that residual-only isolation is conservative. No
selection or longer schedule is allowed before the matched Nynorsk report.

Canonical Nynorsk evaluation completes the comparison:

| Canonical Nynorsk metric | Bundle-32 control | Direct loss | Isolated loss |
| --- | ---: | ---: | ---: |
| UPOS F1 | 98.6720% | **98.6752%** | 98.6624% |
| UFeats F1 | 93.6384% | **94.4672%** | 93.8208% |
| Lemmas F1 | 98.5344% | 98.5056% | **98.5440%** |
| Rare morphology micro F1 | 91.3736% | **92.3979%** | 91.4515% |
| OOV morphology micro F1 | 88.4669% | **89.2680%** | 88.5769% |
| Rare lemma end-to-end | **96.9494%** | 96.9076% | 96.8241% |
| OOV lemma end-to-end | 91.6404% | 91.5615% | **91.8770%** |

Against Bundle-32, the isolated candidate gains 0.1824 UFeats points, 0.0096
Lemmas points, 0.0779/0.1100 Rare/OOV morphology points, and 0.2366 OOV-lemma
points. It loses 0.0096 overall UPOS points, 0.0418/0.0394 Rare/OOV UPOS
points, and 0.1253 Rare-lemma points. These losses correspond to only three,
one, one, and three additional errors respectively.

The isolated checkpoint is selected as the compact reference because it
improves UFeats, Lemmas, Rare/OOV morphology, and OOV lemma on both standards
without a material regression. The unisolated model remains only a morphology
upper control. Epoch 12 won again, so the controlled 30-epoch convergence run
is now permitted. Both official test splits remain untouched.

The predeclared convergence command is:

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-direct-loss-isolated-w010-e30-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --categorical-distillation-objective dkd \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --morphology-bundle-candidate-count 32 \
  --morphology-bundle-loss-weight 0.1 \
  --isolate-morphology-bundle-loss-gradient \
  --early-stopping-patience 4 \
  --epoch-count 30 \
  --morphology-weight-cap 10.0
```

Changing the maximum epoch count also lengthens the linear warmup/decay
schedule. This is therefore a documented convergence ablation rather than a
strict continuation of the 12-epoch optimizer trajectory.

The convergence training run completed in approximately 2 hours 26 minutes
51 seconds. Epoch 12 again produced the lowest combined Development loss, and
patience 4 stopped training after epoch 16. The selected joint signals are:

| Joint Development signal | 30-epoch-schedule checkpoint |
| --- | ---: |
| Combined loss | 0.117648 |
| UPOS accuracy | 98.8273% |
| Lemma-rule accuracy | 98.8293% |
| Bundle loss | 0.132431 |
| Bundle candidate coverage | 98.2180% |

Epochs 13 through 16 did not improve the selection loss. Early stopping
therefore avoided the remaining 14 configured epochs, and the longer schedule
did not move the selected checkpoint beyond epoch 12. This is not yet a
quality decision: lengthening the schedule changed the learning-rate
trajectory, so canonical Bokmål and Nynorsk evaluation against the fixed
12-epoch isolated reference remains required. No UFeats, Lemmas, UPOS, or
Rare/OOV improvement is claimed from the joint training signal alone.

Canonical Bokmål evaluation is complete:

| Bokmål metric | Isolated 12-epoch reference | 30-epoch schedule | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | **98.9854%** | 98.9524% | -0.0330 pp |
| UFeats F1 | 96.0351% | **96.1643%** | +0.1292 pp |
| Lemmas F1 | **98.9881%** | 98.9249% | -0.0632 pp |
| Rare UPOS accuracy | **98.5079%** | 98.0952% | -0.4127 pp |
| Rare morphology micro F1 | **95.6446%** | 95.4655% | -0.1791 pp |
| Rare lemma end-to-end | **97.4603%** | 97.1111% | -0.3492 pp |
| OOV UPOS accuracy | **98.2187%** | 98.1831% | -0.0356 pp |
| OOV morphology micro F1 | 94.2055% | **94.5146%** | +0.3091 pp |
| OOV lemma end-to-end | **93.6587%** | 93.4806% | -0.1781 pp |

The longer schedule recovers 47 additional exact UFeats bundles but loses
12 UPOS and 23 Lemma predictions over the complete Bokmål Development split.
Its remaining UFeats gap to UDPipe narrows from 2.0347 to 1.9055 points, while
the previous 0.0137-point Lemmas lead becomes a 0.0495-point deficit and the
UPOS lead narrows to 0.0027 points. The candidate therefore already fails the
all-task selection gate on Bokmål. Nynorsk evaluation remains necessary to
complete the convergence diagnosis, but it cannot retroactively make this the
shared selected checkpoint.

Canonical Nynorsk evaluation completes the convergence ablation:

| Nynorsk metric | Isolated 12-epoch reference | 30-epoch schedule | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | 98.6624% | **98.6816%** | +0.0192 pp |
| UFeats F1 | 93.8208% | **93.8720%** | +0.0512 pp |
| Lemmas F1 | 98.5440% | **98.5536%** | +0.0096 pp |
| Rare UPOS accuracy | **98.2867%** | 98.1613% | -0.1254 pp |
| Rare morphology micro F1 | **91.4515%** | 91.1543% | -0.2972 pp |
| Rare lemma end-to-end | 96.8241% | **96.8659%** | +0.0418 pp |
| OOV UPOS accuracy | 97.5158% | 97.5158% | 0.0000 pp |
| OOV morphology micro F1 | **88.5769%** | 88.5432% | -0.0336 pp |
| OOV lemma end-to-end | **91.8770%** | 91.6798% | -0.1972 pp |

The longer schedule gains 6 UPOS, 16 complete UFeats bundles, and 3 Lemma
predictions over the complete Nynorsk Development split. These small overall
gains do not compensate for its Rare/OOV regressions or its already failed
Bokmål gate. The 30-epoch convergence candidate is rejected. The
twelve-epoch isolated direct-bundle checkpoint remains the compact Norwegian
reference, and training duration is no longer the next morphology
bottleneck. The next predeclared work is the complete all-feature UDPipe
diagnostic followed, if justified, by a morphology-specific middle gradient
scope.

### All-feature UDPipe diagnostic

The comparison after the 30-epoch gate must not reduce morphology quality to
one aggregate ranking. Official `UFeats` remains the strict complete-bundle
metric, but it cannot identify whether Prism or UDPipe is stronger for a
particular feature. The evaluator now accepts an aligned CoNLL-U prediction
through `--morphology-feature-comparison` and compares the selected checkpoint
with the persisted UDPipe 2.17 prediction during the same model forward pass.
Identical sentence counts, token counts, and token forms are validated before
the report can complete.

The JSON report contains, for every shared feature and both systems:

- overall exact accuracy, including correct absence;
- annotated-token exact accuracy, excluding unannotated `<NONE>` cases;
- per-value support, true/false-positive and false-negative counts, precision,
  recall, and F1;
- Rare/OOV slices using Prism's training-derived frequency classes;
- the contribution of that feature to wrong complete bundles;
- separate canonical and explicit treebank-policy results.

The console prints separate overall/annotated and Rare/OOV tables. Their
columns are explicitly labeled `Prism`, `UDPipe`, and `Prism-UDPipe` by
default; `--morphology-feature-comparison-name` can name another aligned
system without changing the generic comparison core. The full count-level
result remains in the analysis JSON together with that display name.

Run the canonical Bokmål report with:

```shell
.venv/bin/python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nb \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-direct-loss-isolated-w010-e12-weighted/best.pt \
  --analysis runs/no-student-character-cnn-dkd-bundle32-direct-loss-isolated-w010-e12-weighted/nb-development-feature-comparison.json \
  --morphology-logit-correction-strength 1.0 \
  --morphology-feature-comparison runs/udpipe-2.17-251125/ud-2.17/nb-development.conllu
```

The canonical Bokmål report is complete for the selected twelve-epoch
isolated direct-bundle checkpoint:

| Bokmål feature | Prism overall | UDPipe overall | Prism annotated | UDPipe annotated |
| --- | ---: | ---: | ---: | ---: |
| Gender | 97.2477% | **98.7764%** | 92.3788% | **97.1043%** |
| Number | 98.8644% | **99.1614%** | 98.0073% | **98.4982%** |
| Definite | 99.2604% | **99.4418%** | 98.5824% | **98.7866%** |
| Degree | 99.6618% | **99.7195%** | **97.7679%** | 97.6992% |
| VerbForm | 99.6481% | **99.6865%** | 98.8266% | **99.0222%** |

Prism produces 1,442 wrong complete bundles and UDPipe 702. Across those
bundles, Prism has 2,159 individual feature errors and UDPipe 1,387, an excess
of 772. `Gender` contributes 556 of those excess errors, `Number` 108, and
`Definite` 66. Together they explain 730, or 94.6%, of the excess individual
feature errors. `Gender` is the dominant bottleneck on both annotated tokens
(-4.7255 percentage points) and the Rare slice (-5.2063 points).

The comparison also rejects the idea that UDPipe is uniformly stronger.
Prism leads overall exact accuracy for `Case`, `Mood`, `NumType`, `Poss`, and
`Tense`, and ties `Reflex`. On OOV tokens Prism leads nine features, ties four,
and trails five. The architectural follow-up should therefore remain narrowly
morphology-specific and preserve the already stronger features rather than
replace the schema-driven decoder wholesale.

The corresponding canonical Nynorsk report is also complete:

| Nynorsk feature | Prism overall | UDPipe overall | Prism annotated | UDPipe annotated |
| --- | ---: | ---: | ---: | ---: |
| Gender | 95.3504% | **96.3584%** | 86.8156% | **89.8527%** |
| Number | 98.8512% | **99.5424%** | 97.3710% | **97.4043%** |
| Definite | 99.3312% | **99.6064%** | **98.9982%** | 98.8980% |
| Degree | **99.6800%** | 99.6480% | **98.6613%** | 97.8093% |
| Mood | **99.9296%** | 99.8688% | **99.8287%** | 99.2861% |

Prism has 1,931 wrong Nynorsk bundles versus UDPipe's 1,356. Its 2,496
individual feature errors exceed UDPipe's 1,909 by 587. `Gender`, `Number`,
and `Definite` contribute excesses of 315, 216, and 86 respectively. Their
combined 617 errors exceed the entire net deficit because Prism recovers 30
errors elsewhere, principally through `Mood`, `Tense`, `Degree`, and
`VerbForm`.

The Nynorsk `Number` and `Definite` overall deficits are predominantly output
conventions rather than annotated-value recognition. `Number=Sing` has only
two gold occurrences, but canonical Prism emits it 221 times while UDPipe
emits it once. `Definite=Def` has no gold support, but canonical Prism emits it
54 times while UDPipe emits it once; on actually annotated `Definite` tokens,
Prism is slightly better. The existing explicit Nynorsk treebank policy owns
these suppressions. They must not be baked into the shared Norwegian neural
representation merely to improve this benchmark.

`Gender` remains the shared neural bottleneck. Prism is better than UDPipe on
Nynorsk `Gender=Com` recall (2.18% versus 0%), but UDPipe recognizes
`Fem`, `Masc`, and `Neut` more reliably. Together with the Bokmål result, this
authorizes the morphology-specific middle-gradient ablation while requiring
the existing annotation-policy boundary and the stronger feature heads to
remain protected.

This is an architectural diagnostic, not a new selection metric optimized in
isolation. Bokmål confirms that the largest deficits remain `Gender`,
`Number`, and `Definite` for the selected isolated checkpoint, while Prism
already leads several other features. Nynorsk confirms the same concentration
and separates its treebank-convention component from neural quality. The
planned morphology-specific middle-gradient ablation is therefore authorized.
A future candidate is selected only if it improves the joint canonical
Bokmål/Nynorsk all-task gate; merely moving closer to UDPipe on one aggregate
column is insufficient. Both official test splits remain untouched.

### Morphology-scoped direct-bundle gradient

The authorized middle-gradient ablation is implemented as the third explicit
scope between the historical `full` control and selected `residual-only`
reference. `morphology` gives the direct complete-bundle objective gradients
through the morphology adapter, independent feature heads, structured
decoder, and bundle residual projection. It detaches Backbone/shared token
states and UPOS evidence and never enters the lemma path. Normal supervised
and distillation gradients are unchanged.

All scopes use numerically identical candidate scores. Focused tests verify
the exact parameter boundary with dropout active. The old isolation flag
continues to resolve to `residual-only`; the typed scope string is stored in
new checkpoint metadata and does not change model parameters or inference.

The fixed training command is:

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-w010-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --categorical-distillation-objective dkd \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --morphology-bundle-candidate-count 32 \
  --morphology-bundle-loss-weight 0.1 \
  --morphology-bundle-loss-gradient-scope morphology \
  --early-stopping-patience 4 \
  --epoch-count 12 \
  --morphology-weight-cap 10.0
```

The predeclared acceptance gate requires the loss-selected checkpoint to be
evaluated separately on canonical Bokmål and Nynorsk. It includes complete
UPOS/UFeats/Lemmas, per-feature, and Rare/OOV reports; the untouched test
splits remain out of scope.

The fixed training run completed in approximately 2 hours 11 minutes 35
seconds and selected epoch 12. The checkpoint is
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-w010-e12-weighted/best.pt`
and stores `morphology` as its resolved gradient scope. Its 70,068,462-byte
size is identical to the residual-only reference, confirming that the scope
changes training only.

| Joint Development signal | `full` | `residual-only` | `morphology` |
| --- | ---: | ---: | ---: |
| Combined loss | **0.112011** | 0.113910 | 0.112124 |
| UPOS loss | 0.046595 | 0.046849 | **0.046252** |
| Morphology loss | **0.008133** | 0.008676 | 0.008535 |
| Lemma-rule loss | 0.046031 | 0.045161 | **0.044659** |
| Bundle loss | **0.112506** | 0.132241 | 0.126770 |
| UPOS accuracy | 98.8480% | 98.8361% | **98.8672%** |
| Lemma-rule accuracy | 98.7938% | 98.8618% | **98.8648%** |

The result has the intended intermediate shape. Bundle and morphology losses
move toward the stronger `full` morphology control, while lemma loss improves
even beyond `residual-only`; UPOS also improves. The combined loss is only
0.000113 above `full` and 0.001786 below `residual-only`. These joint signals
justify canonical evaluation, but they do not select the model: only separate
Bokmål and Nynorsk UPOS/UFeats/Lemmas, per-feature, and Rare/OOV reports can
show whether the protected gradient boundary improves real predictions.

The canonical Bokmål report, with the predeclared checkpoint-derived logit
correction at strength `1.0`, is now complete:

| Bokmål Development metric | `full` | `residual-only` | `morphology` | UDPipe 2.17 |
| --- | ---: | ---: | ---: | ---: |
| UPOS F1 | 98.9964% | 98.9854% | **98.9991%** | 98.9497% |
| UFeats F1 | **96.7390%** | 96.0351% | 96.1588% | 98.0698% |
| Lemmas F1 | 98.8974% | **98.9881%** | 98.9634% | 98.9744% |
| Rare morphology micro F1 | **96.2849%** | 95.6446% | 95.5501% | — |
| OOV morphology micro F1 | **94.8654%** | 94.2055% | 94.5367% | — |
| Rare lemma end-to-end | 97.1111% | **97.4603%** | 97.4286% | — |
| OOV lemma end-to-end | 92.8393% | **93.6587%** | 93.3025% | — |

Relative to `residual-only`, the middle scope gains 0.1237 UFeats points and
45 additional exact morphology bundles. It also gains 0.3305 OOV-morphology
points and improves UPOS, while giving back 0.0945 Rare-morphology points,
0.0247 Lemmas points, and 0.3562 OOV-lemma points. It therefore behaves as the
intended compromise, but remains 0.5802 UFeats points below the `full` control
and 1.9110 points below UDPipe.

The feature report still identifies `Gender` as the dominant Bokmål deficit:
Prism trails UDPipe by 1.4490 overall points and 4.6989 annotated-token points.
Against `residual-only`, the middle scope improves Gender overall by 0.0797
points and Gender OOV by 0.9619 points, but Rare Gender is unchanged. Number
and Definite remain smaller deficits. This is a real but insufficient recovery
of the morphology signal, so Bokmål alone does not authorize replacing the
selected reference. The matching canonical Nynorsk report remains the
predeclared final gate.

The checkpointed morphology class weights are byte-for-byte numerically equal
across `full`, `residual-only`, and `morphology`. The gradient-scope ablation
therefore does not alter the correction inputs. Strength `1.0` remains fixed
for this comparison; retuning it on Bokmål would confound the scope ablation
and turn the output policy into a checkpoint-specific Development fit.

The matching canonical Nynorsk report completes the gate:

| Nynorsk Development metric | `full` | `residual-only` | `morphology` | UDPipe 2.17 |
| --- | ---: | ---: | ---: | ---: |
| UPOS F1 | 98.6752% | 98.6624% | **98.7136%** | 98.5728% |
| UFeats F1 | **94.4672%** | 93.8208% | 93.9104% | 95.6608% |
| Lemmas F1 | 98.5056% | 98.5440% | **98.5856%** | 98.8288% |
| Rare morphology micro F1 | **92.3979%** | 91.4515% | 91.2468% | — |
| OOV morphology micro F1 | **89.2680%** | 88.5769% | 88.5645% | — |
| Rare lemma end-to-end | 96.9076% | 96.8241% | **97.1166%** | — |
| OOV lemma end-to-end | 91.5615% | 91.8770% | **91.9164%** | — |

Against `residual-only`, the middle scope gains 0.0512 UPOS points, 0.0896
UFeats points, and 0.0416 Lemmas points, corresponding to 16, 28, and 13
additional correct Development predictions. It also improves Rare and OOV
lemma end-to-end by 0.2925 and 0.0394 points. The tradeoff is a 0.2047-point
Rare-morphology regression and a 0.0124-point OOV-morphology regression.
Rare/OOV UPOS also falls by 0.1254/0.0789 points despite the overall UPOS gain.

Across both written standards, `morphology` gains 73 exact UFeats bundles, 21
UPOS predictions, and a net four Lemma predictions over `residual-only`.
However, Rare morphology regresses on both standards, all four Rare/OOV UPOS
slices regress, and Bokmål OOV lemma loses ten correct predictions. Those
task-specific slices initially suggested rejecting the candidate, but did not
yet measure the target quantity: exact complete-bundle UFeats within each
slice. Selection therefore remains provisional until that metric is compared
for both checkpoints on both written standards. The official test splits
remain untouched.

After adding official-compatible metrics to the frequency slices, the
standalone Bokmål rerun measures 88.7619% exact Rare UFeats and 87.3174% exact
OOV UFeats for `morphology`. This corresponds to 354 wrong bundles among 3,150
Rare tokens and 356 among 2,807 OOV tokens. Together the two disjoint slices
contain only 16.38% of Development tokens but account for 710 of the model's
1,397 wrong bundles, or 50.82%. Their morphology micro F1 values of 95.5501%
and 94.5367% are much higher because micro F1 does not invalidate a complete
token when one feature is wrong.

The matched `residual-only` Bokmål rerun reaches 89.3016% Rare UFeats and
85.9637% OOV UFeats. The middle scope therefore loses 17 exact Rare bundles
but gains 38 exact OOV bundles. Of its 45-bundle overall gain, 38 come from
OOV, 24 from forms seen more than five times, and Rare contributes -17. This
directly disproves the preliminary inference that its UFeats gain is
concentrated mainly in frequent tokens. It makes a meaningful OOV trade:
+1.3537 UFeats points against -0.1425 UPOS points and -0.3562 Lemmas points.
The candidate rejection is withdrawn pending the equivalent exact Nynorsk
slice comparison.

The standalone Nynorsk `morphology` report records 86.3769% exact Rare UFeats
and 83.3202% exact OOV UFeats. That is 326 wrong complete bundles among 2,393
Rare tokens and 423 among 2,536 OOV tokens. Together these slices are 15.77%
of Nynorsk Development tokens but contribute 749 of the candidate's 1,903
wrong bundles, or 39.36%.

The matched `residual-only` Nynorsk report reaches 86.2934% Rare UFeats and
83.3596% OOV UFeats. The middle scope therefore gains two exact Rare bundles
and loses one exact OOV bundle. Across both written standards it gains 37 OOV
bundles and loses 15 Rare bundles, while the complete splits gain 73 UFeats,
21 UPOS, and a net four Lemma predictions. On OOV tokens specifically, that
is +37 UFeats predictions against -6 UPOS and -9 Lemmas. Since exact UFeats is
the demonstrated remaining system-level deficit, OOV generalization is the
more important real-world guardrail, five of six complete-split task metrics
improve, and model size/inference are unchanged, `morphology` passes the joint
gate and replaces `residual-only` as the compact Norwegian reference. The
Rare regression remains explicitly recorded rather than hidden by the
selection. `residual-only` remains the protected-gradient control and `full`
the morphology upper control. The official test splits remain untouched.

The selected reference now compares with reproduced UDPipe 2.17 as follows:

| Development metric | Prism `morphology` | UDPipe 2.17 | Prism - UDPipe |
| --- | ---: | ---: | ---: |
| Bokmål UPOS F1 | **98.9991%** | 98.9497% | +0.0494 pp |
| Bokmål UFeats F1 | 96.1588% | **98.0698%** | -1.9110 pp |
| Bokmål Lemmas F1 | 98.9634% | **98.9744%** | -0.0110 pp |
| Nynorsk UPOS F1 | **98.7136%** | 98.5728% | +0.1408 pp |
| Nynorsk UFeats F1 | 93.9104% | **95.6608%** | -1.7504 pp |
| Nynorsk Lemmas F1 | 98.5856% | **98.8288%** | -0.2432 pp |

Prism leads UPOS on both written standards and is effectively tied on Bokmål
Lemmas. UFeats remains the dominant gap: the selected scope narrows the prior
`residual-only` deficit by 0.1237 points on Bokmål and 0.0896 points on
Nynorsk. Nynorsk Lemmas remains the second material deficit.

The standalone evaluator now preserves exact per-feature correct counts for
Rare/OOV slices and ranks their feature errors. With the same selected
checkpoint, canonical output, and fixed correction strength `1.0`, the OOV
attribution is:

| OOV feature | Bokmål errors | Bokmål share | Nynorsk errors | Nynorsk share |
| --- | ---: | ---: | ---: | ---: |
| Gender | 251 | 46.65% | 312 | 53.52% |
| Number | 99 | 18.40% | 96 | 16.47% |
| Definite | 61 | 11.34% | 63 | 10.81% |
| VerbForm | 46 | 8.55% | 38 | 6.52% |
| Degree | 40 | 7.43% | 46 | 7.89% |

These five features account for 92.37% of Bokmål and 95.21% of Nynorsk OOV
feature errors. The denominator is the sum of wrong feature decisions, not
wrong complete bundles: a token with wrong `Gender` and `Number` contributes
two errors here but only one official UFeats failure. The agreement across
both standards makes `Gender` the strongest shared OOV bottleneck, followed
by `Number`; it does not by itself prove that a larger classifier head is the
remedy.

For new Norwegian CLI runs, enabling a positive
`--morphology-bundle-loss-weight` without an explicit gradient scope now
resolves to `morphology`. Loss weight zero still resolves to `full` because no
bundle gradient exists. Explicit `full`, `morphology`, `residual-only`, and
the legacy isolation alias remain available for reproducible ablations.

## Character-aware format-3 distilled Student

The controlled distillation run uses the accepted format-3 Teacher, a fresh
xsmall Student, temperature 1.0, distillation weight 0.1, and the otherwise
unchanged twelve-epoch gold-only policy. Development-loss selection chose
epoch 8:

- Checkpoint: `runs/no-student-character-cnn-distilled-w010-t100-e12-weighted/best.pt`
- Selected checkpoint: epoch 8 of 12
- Joint development loss: 0.109941
- Gold-only control joint development loss: 0.112245
- Checkpoint size: 69,863,132 bytes
- End-to-end wall time: approximately 1 hour 34 minutes 9 seconds

The joint loss is 0.002304 lower than the fixed gold-only control and is a
promising training signal. Separate Bokmål/Nynorsk and Rare/OOV evaluations
remain mandatory before accepting or rejecting distillation. The official test
splits remain untouched.

Temperature 1.0 and weight 0.1 are also the CLI defaults now because the
historical controlled ablation selected this policy; the rejected 2.0/0.5
combination must not remain the implicit starting point.

Bokmål shows a small mixed result:

| Bokmål metric | Gold-only Student | Distilled Student | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.086190 | **0.084777** | -0.001413 |
| Overall UPOS accuracy | 98.9469% | 98.9469% | 0.0000 pp |
| Overall lemma-rule accuracy | 98.8223% | **98.8415%** | +0.0192 pp |
| Rare UPOS accuracy | **98.3175%** | 98.2222% | -0.0953 pp |
| Rare lemma end-to-end accuracy | 96.8889% | **97.0794%** | +0.1905 pp |
| Rare morphology micro F1 | **93.6829%** | 93.6418% | -0.0411 pp |
| OOV UPOS accuracy | 98.2187% | 98.2187% | 0.0000 pp |
| OOV lemma end-to-end accuracy | 92.9106% | 92.9106% | 0.0000 pp |
| OOV morphology micro F1 | 92.6637% | **92.7381%** | +0.0744 pp |

The lower loss and lemma gains are positive, but the effect is too small and
mixed to select distillation from Bokmål alone. Nynorsk remains the second
predeclared acceptance half.

Nynorsk improves on every reported comparison metric:

| Nynorsk metric | Gold-only Student | Distilled Student | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.142569 | **0.139228** | -0.003341 |
| Overall UPOS accuracy | 98.5664% | **98.5920%** | +0.0256 pp |
| Overall lemma-rule accuracy | 98.5364% | **98.5812%** | +0.0448 pp |
| Rare UPOS accuracy | 98.2031% | **98.2449%** | +0.0418 pp |
| Rare lemma end-to-end accuracy | 96.8659% | **96.9076%** | +0.0417 pp |
| Rare morphology micro F1 | 88.2860% | **88.4933%** | +0.2073 pp |
| OOV UPOS accuracy | 97.3580% | **97.4763%** | +0.1183 pp |
| OOV lemma end-to-end accuracy | 90.8123% | **91.1672%** | +0.3549 pp |
| OOV morphology micro F1 | 85.7484% | **85.7792%** | +0.0308 pp |

The distilled Student is selected as the new compact reference. Across both
written standards it lowers loss and improves lemma accuracy; Bokmål overall
UPOS is unchanged and Nynorsk UPOS improves. No OOV metric regresses. The
selection keeps the small Bokmål Rare-UPOS and Rare-morphology regressions of
0.0953 and 0.0411 percentage points explicit. The gain is controlled but much
smaller than the Teacher headroom, so future distillation work should be
task-specific rather than another blind global temperature/weight sweep.

### Rejected task-specific distillation ablation

Prism supports independent temperatures and loss weights for UPOS, morphology,
and lemma. The first predeclared candidate changed only the three task weights
relative to the selected uniform-policy Student:

| Task | Temperature | Selected reference weight | Candidate weight |
| --- | ---: | ---: | ---: |
| UPOS | 1.0 | 0.10 | 0.05 |
| Morphology | 1.0 | 0.10 | 0.20 |
| Lemma rules | 1.0 | 0.10 | 0.10 |

The run used the same data, seed, architecture, optimizer, twelve-epoch
schedule, Teacher checkpoint, and development selection as the accepted
reference. Epoch 8 was selected after approximately 1 hour 35 minutes 21
seconds. Its 69,863,260-byte checkpoint has joint development loss 0.109877,
only 0.000064 below the uniform reference.

```shell
python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-distilled-task-policy-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --distillation-temperature 1.0 \
  --upos-distillation-weight 0.05 \
  --morphology-distillation-weight 0.20 \
  --lemma-rule-distillation-weight 0.10 \
  --morphology-weight-cap 10.0
```

Bokmål is mixed and misses the morphology objective:

| Bokmål metric | Uniform 0.10 | Task-specific | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.084777 | **0.084125** | -0.000652 |
| Overall UPOS accuracy | **98.9469%** | 98.9442% | -0.0027 pp |
| Overall lemma-rule accuracy | **98.8415%** | 98.8250% | -0.0165 pp |
| Rare UPOS accuracy | 98.2222% | **98.3492%** | +0.1270 pp |
| Rare lemma end-to-end accuracy | 97.0794% | 97.0794% | 0.0000 pp |
| Rare morphology micro F1 | **93.6418%** | 93.6274% | -0.0144 pp |
| OOV UPOS accuracy | 98.2187% | 98.2187% | 0.0000 pp |
| OOV lemma end-to-end accuracy | **92.9106%** | 92.6612% | -0.2494 pp |
| OOV morphology micro F1 | **92.7381%** | 92.7156% | -0.0225 pp |

Nynorsk gains morphology but regresses more broadly:

| Nynorsk metric | Uniform 0.10 | Task-specific | Change |
| --- | ---: | ---: | ---: |
| Development loss | **0.139228** | 0.139848 | +0.000620 |
| Overall UPOS accuracy | **98.5920%** | 98.5824% | -0.0096 pp |
| Overall lemma-rule accuracy | **98.5812%** | 98.5588% | -0.0224 pp |
| Rare UPOS accuracy | **98.2449%** | 98.2031% | -0.0418 pp |
| Rare lemma end-to-end accuracy | 96.9076% | **96.9494%** | +0.0418 pp |
| Rare morphology micro F1 | 88.4933% | **88.8060%** | +0.3127 pp |
| OOV UPOS accuracy | 97.4763% | 97.4763% | 0.0000 pp |
| OOV lemma end-to-end accuracy | **91.1672%** | 91.1278% | -0.0394 pp |
| OOV morphology micro F1 | 85.7792% | **85.8340%** | +0.0549 pp |

The candidate is rejected because its Nynorsk morphology gains do not transfer
to Bokmål and come with broader regressions on both written standards. The
uniform temperature-1.0, weight-0.1 Student remains selected. This experiment
tests task-specific logit weighting, not full DKD; the official test splits
remain untouched.

### Selected categorical DKD ablation

The implemented DKD objective separates target-class knowledge (TCKD) from the
renormalized non-target distribution (NCKD) for categorical outputs. It applies
to UPOS, lemma rules, and exclusive morphology features. Multi-value
morphology remains on binary KL because it has no single target class.

The first candidate isolates this objective change. It restores the selected
uniform temperature 1.0 and outer task weight 0.1, then assigns both TCKD and
NCKD component weight 1.0. Data, seed, model architecture, optimizer,
twelve-epoch schedule, Teacher, checkpoint selection, and evaluation remain
fixed. Development-loss selection chose epoch 12 after approximately 1 hour
41 minutes 49 seconds:

- Checkpoint: `runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/best.pt`
- Joint development loss: 0.101139
- Uniform-KL reference joint development loss: 0.109941
- Checkpoint size: 69,863,388 bytes

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --categorical-distillation-objective dkd \
  --dkd-target-class-weight 1.0 \
  --dkd-non-target-class-weight 1.0 \
  --morphology-weight-cap 10.0
```

Separate Bokmål evaluation favors DKD:

| Bokmål metric | Uniform KL | DKD | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.084777 | **0.077811** | -0.006966 |
| Overall UPOS accuracy | 98.9469% | **98.9634%** | +0.0165 pp |
| Overall lemma-rule accuracy | 98.8415% | **99.0231%** | +0.1816 pp |
| Rare UPOS accuracy | 98.2222% | **98.3810%** | +0.1588 pp |
| Rare lemma end-to-end accuracy | 97.0794% | **97.5238%** | +0.4444 pp |
| Rare morphology micro F1 | 93.6418% | **94.4867%** | +0.8449 pp |
| OOV UPOS accuracy | **98.2187%** | 98.0406% | -0.1781 pp |
| OOV lemma end-to-end accuracy | 92.9106% | **93.0887%** | +0.1781 pp |
| OOV morphology micro F1 | 92.7381% | **93.3333%** | +0.5952 pp |

Nynorsk independently confirms the broader gain:

| Nynorsk metric | Uniform KL | DKD | Change |
| --- | ---: | ---: | ---: |
| Development loss | 0.139228 | **0.128288** | -0.010940 |
| Overall UPOS accuracy | 98.5920% | **98.6368%** | +0.0448 pp |
| Overall lemma-rule accuracy | 98.5812% | **98.6357%** | +0.0545 pp |
| Rare UPOS accuracy | **98.2449%** | 98.1195% | -0.1254 pp |
| Rare lemma end-to-end accuracy | 96.9076% | **97.0748%** | +0.1672 pp |
| Rare morphology micro F1 | 88.4933% | **89.4502%** | +0.9569 pp |
| OOV UPOS accuracy | **97.4763%** | 97.3975% | -0.0788 pp |
| OOV lemma end-to-end accuracy | 91.1672% | **91.6009%** | +0.4337 pp |
| OOV morphology micro F1 | 85.7792% | **86.7135%** | +0.9343 pp |

DKD is selected as the new compact Student reference. It improves joint and
per-standard loss, overall UPOS and lemma, and Rare/OOV lemma and morphology
on both written standards without changing inference cost. The smaller OOV
UPOS regressions on both standards and the Nynorsk Rare-UPOS regression remain
explicit tradeoffs. The official test splits remain untouched.

## Gold-tokenized UDPipe 2.17 comparison

This comparison uses the official word-level CoNLL evaluation definitions for
`UPOS`, complete universal `UFeats`, and `Lemmas`. It does **not** compare
Prism's easier per-feature morphology accuracies with UDPipe `UFeats`: every
universal feature on a word must match for that word to count as correct.
Because both systems receive gold tokenization, precision, recall, F1, and
aligned accuracy are numerically identical; JSON retains all counts and all
four values.

The exact UD 2.17 data references are:

- Bokmål `r2.17` / `b8618a2b935762d6ccd2dc997180c3e46f74f6b7`;
- Nynorsk `r2.17` / `2bbe9c67d5e81eadf237b7840ebac31bffca38ae`;
- both treebanks: CC BY-SA 4.0.

SHA-256 comparison confirms that their train, development, and test CoNLL-U
files are byte-identical to the currently pinned Norwegian files. The selected
DKD Student therefore trained on the exact same train content and is evaluated
on the exact same development content used here. The test files were hashed
only to establish dataset identity; neither test split was evaluated.

UDPipe uses the official service models
`norwegian-bokmaal-ud-2.17-251125` and
`norwegian-nynorsk-ud-2.17-251125`. UDPipe predictions are persisted under
`runs/udpipe-2.17-251125/ud-2.17/` and the local score implementation was
cross-checked against the official updated CoNLL-2018 evaluator on all 36,369
Bokmål development tokens. UPOS, UFeats, and Lemmas matched exactly to full
floating-point precision.

| Development F1 | Prism DKD Student | UDPipe 2.17 | Prism change |
| --- | ---: | ---: | ---: |
| Bokmål UPOS | **98.9634%** | 98.9497% | +0.0137 pp |
| Bokmål UFeats | 93.9151% | **98.0698%** | -4.1547 pp |
| Bokmål Lemmas | 98.9469% | **98.9744%** | -0.0275 pp |
| Nynorsk UPOS | **98.6368%** | 98.5728% | +0.0640 pp |
| Nynorsk UFeats | 91.5232% | **95.6608%** | -4.1376 pp |
| Nynorsk Lemmas | 98.5536% | **98.8288%** | -0.2752 pp |

The result is specific and actionable: the compact shared Prism model already
matches or narrowly exceeds UDPipe on UPOS and is close on lemma, while exact
morphology bundles remain substantially behind. The next quality work should
therefore target joint morphology consistency rather than broadly enlarging
every task.

### UFeats error concentration and logit-correction ablation

Per-feature exact comparison localizes most of the complete-bundle deficit:

| Feature accuracy | Prism Bokmål | UDPipe Bokmål | Prism Nynorsk | UDPipe Nynorsk |
| --- | ---: | ---: | ---: | ---: |
| Gender | 95.4467% | 98.7764% | 93.4784% | 96.3584% |
| Number | 98.6857% | 99.1614% | 98.5376% | 99.5424% |
| Definite | 99.0954% | 99.4418% | 99.1936% | 99.6064% |

The selected Student makes 3,007 individual feature errors across 2,213 wrong
Bokmål bundles and 3,364 feature errors across 2,649 wrong Nynorsk bundles.
That is only 1.36 and 1.27 wrong features per incorrect bundle on average.
Gender alone contributes 1,656 Bokmål and 2,038 Nynorsk feature errors.

Nynorsk also reveals an annotation-policy conflict in the shared raw target
space. `Gender=Com` recall is 3.27% over 733 gold tokens, while `Fem` and `Masc`
receive 1,314 and 1,291 false positives. `Number=Sing` has only two gold tokens
but 285 false positives; `Definite=Def` has no gold support but 64 false
positives. These must be separated into linguistic model quality and external
treebank-output convention before changing model capacity.

The completed predeclared, no-retraining ablation evaluates correction
strengths `0.0`, `0.25`, `0.5`, `0.75`, and `1.0`. For each morphology logit
`z` and its checkpointed training class weight `w`, evaluation uses
`z - strength * log(w)`. Development loss stays raw and comparable; exact
UFeats, per-feature metrics, label precision/recall/AP, and Rare/OOV summaries
use the corrected prediction logits.

| Strength | Bokmål UFeats | Bokmål Rare F1 | Bokmål OOV F1 | Nynorsk UFeats | Nynorsk Rare F1 | Nynorsk OOV F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 93.9151% | 94.4867% | 93.3333% | 91.5232% | 89.4502% | 86.7135% |
| 0.25 | 94.7070% | 94.9243% | 93.8073% | 92.0064% | 90.0961% | 87.1035% |
| 0.50 | 95.3202% | 95.1857% | 94.0294% | 92.5376% | 90.4159% | 87.6099% |
| 0.75 | 95.7271% | 95.4009% | **94.1721%** | 93.0720% | 90.5719% | 88.0380% |
| **1.00** | **95.7601%** | **95.4817%** | 94.0673% | **93.3920%** | **90.7920%** | **88.3686%** |

Full correction closes 1.8450 of the original 4.1547 percentage-point Bokmål
gap and 1.8688 of the 4.1376-point Nynorsk gap to UDPipe, or 44.4% and 45.2%.
Strength `1.0` is selected as the shared Norwegian output policy because it
maximizes exact UFeats on both standards and clearly improves the Nynorsk
Rare/OOV slices. Bokmål OOV morphology peaks at `0.75`, but its 0.1048-point
advantage over `1.0` is smaller than the joint exact-UFeats and Nynorsk gains.
The evaluator keeps zero as its backward-compatible CLI default; the selected
release artifact will carry `1.0` and its derived correction offsets
explicitly. Both test splits remain untouched.

The tensor-only export adapters now embed these fixed offsets as registered
buffers and subtract them inside the exported graph. Strict eager/export
parity covers the selected character-aware architecture, removing any need for
Swift or C++ to reproduce the correction formula. Production manifest wiring
and backend-specific artifact generation remain separate release work.

### Treebank-annotation policy ablation

The next reversible ablation separates canonical Prism morphology from the
external UD treebank convention. It is enabled only with
`--ud-morphology-policy treebank`; `canonical` remains the default. The policy
maps combined feminine/masculine gender to `Com` for adjectives and
determiners. For Nynorsk it additionally removes `Number=Sing` and
`Definite=Def`, following its pinned training convention. Only official UFeats
uses the mapped output. Per-label, Average Precision, and Rare/OOV reports
remain canonical and therefore directly comparable with the preceding logit
grid.

The fixed Development commands are:

```shell
.venv/bin/python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nb \
  --checkpoint runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/best.pt \
  --analysis runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/nb-development-analysis-ud-policy.json \
  --morphology-logit-correction-strength 1.0 \
  --ud-morphology-policy treebank

.venv/bin/python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nn \
  --checkpoint runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/best.pt \
  --analysis runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/nn-development-analysis-ud-policy.json \
  --morphology-logit-correction-strength 1.0 \
  --ud-morphology-policy treebank
```

No result or selection is recorded before both commands complete. Both test
splits remain untouched.

Both commands are now complete:

| Development UFeats F1 | Canonical + correction | Treebank policy | UDPipe 2.17 |
| --- | ---: | ---: | ---: |
| Bokmål | 95.7601% | 95.7601% | **98.0698%** |
| Nynorsk | 93.3920% | **95.9136%** | 95.6608% |

The policy is neutral on Bokmål. On Nynorsk it adds 2.5216 percentage points
over the selected corrected canonical output and leads UDPipe by 0.2528
points. Relative to the original uncorrected 91.5232% Student output, the
combined correction and annotation policy add 4.3904 points. Raw per-label,
Average Precision, and Rare/OOV metrics remain identical, confirming that the
gain is isolated to the external complete-bundle annotation contract rather
than a changed model prediction. Both test splits remain untouched.

The evaluation accumulator now emits a sequential marginal audit for those
three named rules. For each step, `changed` counts altered complete bundles,
`improved` counts bundles that become exactly correct, and `regressed` counts
previously correct bundles that become wrong. These counts are included in the
terminal report and JSON. Both commands above must be repeated once to freeze
or reject each component rather than accepting the aggregate results blindly;
an unchanged aggregate can still hide mutually cancelling improvements and
regressions.

The completed audit is:

| Profile and step | Changed bundles | Improved bundles | Regressed bundles |
| --- | ---: | ---: | ---: |
| Bokmål `common-gender` | 5 | 0 | 0 |
| Nynorsk `common-gender` | 730 | 611 | 0 |
| Nynorsk `nynorsk-number` | 239 | 134 | 0 |
| Nynorsk `nynorsk-definite` | 56 | 43 | 0 |

The sequential Nynorsk improvements sum to 788 bundles. Over 31,250
Development tokens this is exactly 2.5216 percentage points, matching the
aggregate UFeats gain. No rule regresses a previously correct bundle. The
external Nynorsk treebank policy is selected and frozen on Development. It
does not replace Prism's canonical morphology for mixed Norwegian input, and
the test splits remain untouched.

### Training-derived morphology-bundle oracle

Before adding a bundle reranker, Prism measures whether a compact candidate
space can contain the correct complete bundle at all. The inventory is built
only from the joint Bokmål/Nynorsk training splits, ranked by training
frequency within each UPOS, and evaluated on the selected Development split.
The reported gold-UPOS oracle deliberately removes UPOS prediction errors so
it measures candidate coverage rather than reranker quality. Annotated-token
coverage excludes the empty bundle.

```shell
.venv/bin/python -m prism.languages.norwegian.analyze_morphology_bundles \
  --language-tag nb \
  --analysis runs/morphology-bundles/nb-development-oracle.json

.venv/bin/python -m prism.languages.norwegian.analyze_morphology_bundles \
  --language-tag nn \
  --analysis runs/morphology-bundles/nn-development-oracle.json
```

The shared training inventory contains 489,216 token examples, 256 distinct
bundles, 298 distinct UPOS-bundle pairs, and a maximum of 74 candidates for a
single UPOS.

| Gold-UPOS oracle coverage | Bokmål all | Bokmål annotated | Nynorsk all | Nynorsk annotated |
| --- | ---: | ---: | ---: | ---: |
| Top 1 | 57.7360% | 28.9252% | 54.5216% | 23.5064% |
| Top 2 | 65.2671% | 41.6647% | 67.6768% | 45.7051% |
| Top 4 | 80.9288% | 68.2257% | 73.8112% | 56.1072% |
| Top 8 | 88.2620% | 80.6622% | 84.9120% | 74.9308% |
| Top 16 | 95.5951% | 93.0986% | 94.0160% | 90.3684% |
| Top 32 | 99.2769% | 98.7736% | 96.9184% | 94.8125% |
| Full inventory | 99.9973% | 99.9953% | 99.5200% | 99.1861% |

Top 4 and top 8 exclude too many legitimate bundles to be safe. Top 32 is a
practical first reranker ceiling: it covers almost every annotated Bokmål
bundle and 94.8125% of annotated Nynorsk bundles while remaining tiny compared
with the neural backbone. Production decoding must still fall back to the
independent feature decoder when a bundle is unseen or the predicted UPOS
candidate set is unsuitable. No test split was evaluated.

The corresponding trainable ablation is implemented and selected. The resolved
joint inventory produces 185 candidates and adds
35,723 trainable parameters, approximately 143 KB of raw FP32 parameter
values. It can be trained without changing the selected architecture enum:

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --categorical-distillation-objective dkd \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --morphology-bundle-candidate-count 32 \
  --morphology-weight-cap 10.0
```

The 12-epoch run selected epoch 12. Its checkpoint is 70,068,078 bytes, versus
69,863,388 bytes for the previous DKD reference. Canonical Development results
against that separately trained reference are:

| Canonical Development metric | DKD reference Bokmål | Bundle-32 Bokmål | Change | DKD reference Nynorsk | Bundle-32 Nynorsk | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UPOS accuracy | 98.9634% | **99.0046%** | +0.0412 pp | 98.6368% | **98.6720%** | +0.0352 pp |
| Lemma-rule accuracy | 99.0231% | **99.0369%** | +0.0138 pp | **98.6357%** | 98.6165% | -0.0192 pp |
| UFeats F1 | 95.7601% | **96.0076%** | +0.2475 pp | 93.3920% | **93.6384%** | +0.2464 pp |
| Morphology micro F1 | 97.6088% | **97.6722%** | +0.0634 pp | 94.7210% | **94.9724%** | +0.2514 pp |
| Rare morphology micro F1 | 95.4817% | **95.5196%** | +0.0379 pp | 90.7920% | **91.3736%** | +0.5816 pp |
| OOV morphology micro F1 | 94.0674% | **94.1716%** | +0.1042 pp | 88.3686% | **88.4669%** | +0.0983 pp |

The small Nynorsk lemma tradeoff is outweighed by consistent UPOS, complete
bundle, micro-F1, and Rare/OOV morphology gains across both written standards.
OOV lemma end-to-end accuracy also improves by 0.5344 points on Bokmål and
0.0394 points on Nynorsk. Bundle-32 is therefore selected on real canonical
model output, not on a benchmark-only annotation conversion.

The `--disable-morphology-bundle-reranker` pass on the same trained checkpoint
confirms that the residual component is active: canonical UFeats falls to
86.9724% on Bokmål and 87.7824% on Nynorsk. It is not an independent quality
control, because the jointly trained independent heads co-adapt to the
reranker. The separately trained DKD checkpoint in the table above remains the
valid selection control.

The final external-policy comparison is:

| Development UFeats F1 | Canonical Bundle-32 | Treebank policy | UDPipe 2.17 | Prism vs UDPipe |
| --- | ---: | ---: | ---: | ---: |
| Bokmål | **96.0076%** | 96.0048% | **98.0698%** | -2.0650 pp |
| Nynorsk | 93.6384% | **96.1920%** | 95.6608% | +0.5312 pp |

The Bokmål `common-gender` policy changes six bundles, improves none, and
regresses one; it must not be presented as a model gain. On Nynorsk,
`common-gender`, `nynorsk-number`, and `nynorsk-definite` improve 618, 138, and
42 bundles respectively with zero regressions. Their 798 improvements exactly
account for the 2.5536-point policy gain over 31,250 tokens. Both test splits
remain untouched.

### Bokmål Gender error audit

The remaining Bokmål gap is concentrated in `Gender`, followed by `Number`
and `Definite`. A token-aligned audit now attributes one selected feature's
errors by confusion, gold/predicted UPOS, training-frequency class, normalized
form, and immediate gold-UPOS context. It also checks an aligned external
CoNLL-U prediction twice: once for the selected feature and once for the full
bundle. This distinction prevents a correct UDPipe Gender value from being
misreported as a completely correct morphology prediction.

Run the first audit on the selected canonical Bundle-32 output:

```shell
.venv/bin/python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nb \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-e12-weighted/best.pt \
  --analysis runs/no-student-character-cnn-dkd-bundle32-e12-weighted/nb-development-gender-audit.json \
  --morphology-logit-correction-strength 1.0 \
  --morphology-error-audit-feature Gender \
  --morphology-error-audit-comparison runs/udpipe-2.17-251125/ud-2.17/nb-development.conllu
```

The complete token records are serialized under `morphology_error_audit`; the
terminal prints the leading aggregates. The completed result is:

| Gender audit slice | Prism errors | UDPipe feature-correct | UDPipe bundle-correct |
| --- | ---: | ---: | ---: |
| All | 1,033 | 772 | 756 |
| Frequent | 503 | 401 | 394 |
| Rare | 261 | 196 | 193 |
| OOV | 269 | 175 | 169 |

The errors are concentrated by gold UPOS in NOUN (561), ADJ (191), and PROPN
(142). The four largest confusions are `Fem -> Masc` (212), `Masc -> <NONE>`
(147), `Neut -> <NONE>` (78), and `<NONE> -> Masc` (74). Half the errors are
Rare/OOV, while frequent ambiguous forms such as `den` and possessive forms
also recur. A hard static lookup would therefore be incomplete and risks
overriding context-dependent analyses.

The checkpoint's candidate metadata provides a second diagnostic without
another model run. The correct full bundle is already among the Top-32
candidates under predicted UPOS for 876 of the 1,033 Gender errors. Of the 756
errors where UDPipe predicts the complete bundle correctly, 672 already expose
the correct candidate to Prism's reranker and only 84 miss it. A wider
inventory cannot close most of the gap; candidate scoring and cross-token
context are now the more plausible bottlenecks.

The next controlled ablation is implemented as a compact gated sentence-level
morphology agreement refiner. It consumes token representations, soft UPOS,
and current morphology evidence, but no gold UPOS or morphology at inference.
The first specification uses a radius of three, excludes the current token,
and refines only `Definite`, `Gender`, and `Number` through a 64-dimensional
bottleneck. It adds 29,707 parameters, approximately 119 KB raw FP32.

Train the matched ablation with:

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-agreement3-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --categorical-distillation-objective dkd \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --morphology-bundle-candidate-count 32 \
  --morphology-agreement-window-radius 3 \
  --morphology-weight-cap 10.0
```

The 12-epoch run selected epoch 10. Its canonical Development comparison is:

| Metric | Bundle-32 Bokmål | Agreement Bokmål | Change | Bundle-32 Nynorsk | Agreement Nynorsk | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UPOS | 99.0046% | **99.0074%** | +0.0028 pp | 98.6720% | **98.6944%** | +0.0224 pp |
| UFeats | **96.0076%** | 95.7821% | -0.2255 pp | 93.6384% | **93.7824%** | +0.1440 pp |
| Lemmas | **98.9607%** | 98.8837% | -0.0770 pp | 98.5344% | **98.5472%** | +0.0128 pp |
| Rare morphology micro F1 | **95.5196%** | 95.4418% | -0.0778 pp | 91.3736% | **91.7564%** | +0.3828 pp |
| OOV morphology micro F1 | **94.1716%** | 93.8937% | -0.2779 pp | **88.4669%** | 88.3450% | -0.1219 pp |

On Bokmål, annotated `Definite`, `Gender`, and `Number` regress by 0.1021,
0.3819, and 0.1319 points respectively. Nynorsk's mixed gains cannot offset a
regression on the other written standard and on OOV morphology. The refiner
therefore fails its target-feature and shared-standard acceptance criteria and
is rejected. Bundle-32 remains selected. This decision uses canonical model
output rather than UDPipe imitation; both test splits remain untouched.

Example for the full correction endpoint:

```shell
.venv/bin/python -m prism.languages.norwegian.evaluate_baseline \
  --language-tag nb \
  --checkpoint runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/best.pt \
  --analysis runs/no-student-character-cnn-dkd-t100-a100-b100-e12-weighted/nb-development-analysis-logit-correction-100.json \
  --morphology-logit-correction-strength 1.0
```

The reproducible external commands are:

```shell
.venv/bin/python -m prism.languages.norwegian.benchmark_udpipe \
  --language-tag nb \
  --treebank-release 2.17

.venv/bin/python -m prism.languages.norwegian.benchmark_udpipe \
  --language-tag nn \
  --treebank-release 2.17
```

Pass `--reuse-prediction` to recompute results without contacting the service.
Future comparison-only training and evaluation use
`--treebank-release 2.17`; the release and exact treebank revisions are stored
in new checkpoints.

### License boundary

Independent Prism training on these Norwegian UD 2.17 treebanks is permitted
under CC BY-SA 4.0, including commercial use, subject to attribution,
license-link, change-notice, and ShareAlike obligations where they apply. The
published UDPipe 2.17 **model weights** are a separate CC BY-NC-SA artifact:
they are suitable as an external benchmark but must not be bundled into or
used as a commercial LexKeep runtime/teacher without separate permission.

Whether trained weights legally constitute adapted material of a dataset can
depend on jurisdiction and facts. Prism therefore keeps source-code,
treebank, backbone, and released-model licensing separate, records complete
provenance, and should obtain legal review before choosing the distribution
license of a commercial model artifact. This project note is not legal advice.

## Historical format-2 NorBERT4-Base teacher

The first historical teacher uses the same supervised data, schema, task heads, and
class-weighting policy as the selected joint student, but replaces the
17-million-parameter xsmall backbone with the 149-million-parameter Base
backbone. The Apache-2.0 checkpoint is pinned at
`386ba2dc5ae5f95fec86d580c5fc4af34d380126`.

- Selected checkpoint: epoch 4
- Checkpoint: `runs/no-teacher-base/best.pt`
- Checkpoint size: 598,665,563 bytes
- End-to-end wall time: approximately 2 hours 20 minutes 31 seconds

| Development metric | Bokmål | Nynorsk |
| --- | ---: | ---: |
| Joint loss | 0.077641 | 0.118087 |
| UPOS accuracy | 99.20% | 98.90% |
| Lemma-rule accuracy | 98.97% | 98.87% |
| Morphology micro precision | 96.17% | 92.66% |
| Morphology micro recall | 98.65% | 97.13% |
| Morphology micro F1 | 97.40% | 94.84% |
| Morphology macro F1 | 96.84% | 90.58% |
| Morphology macro Average Precision | 98.74% | 93.42% |

The teacher exceeds the shared gold-only student on both written standards
and is therefore accepted for the first logit-distillation experiment.
NorBERT4-large remains deferred until Base has demonstrated a measurable gain
in the shipped xsmall student. Both test splits remain untouched.

## Shared Norwegian distilled student

The first distillation ablation keeps the selected NorBERT4-Base teacher
frozen and trains a fresh NorBERT4-xsmall student from the same pretrained
backbone initialization, joint gold data, schema, seed, optimizer policy, and
five-epoch schedule as the gold-only reference. The training objective adds
temperature-scaled logit distillation for UPOS, every morphology feature, and
lemma rules. Checkpoint selection continues to use the supervised combined
development loss.

Two initial policies did not beat the gold-only student:

| Temperature | Distillation weight | Wall time | Result |
| ---: | ---: | ---: | --- |
| 2.0 | 0.5 | 32m 24s | Rejected: distillation dominated the objective and reduced lemma quality and morphology recall. |
| 2.0 | 0.1 | 32m 27s | Rejected: closer to gold-only, but still lower on both written standards. |
| 1.0 | 0.1 | 32m 10s | Selected: first controlled gain over gold-only on both written standards. |

The selected checkpoint is
`runs/no-student-distilled-w010-t100/best.pt`, selected at epoch 5. It is
68,740,059 bytes and records the teacher checkpoint, pinned teacher backbone,
temperature, weight, supervised losses, and distillation losses.

| Development metric | Bokmål gold-only | Bokmål distilled | Nynorsk gold-only | Nynorsk distilled |
| --- | ---: | ---: | ---: | ---: |
| Joint loss | 0.176108 | **0.175509** | 0.216549 | **0.215845** |
| UPOS accuracy | 98.64% | **98.66%** | **98.30%** | 98.29% |
| Lemma-rule accuracy | 96.70% | **96.71%** | 96.64% | **96.65%** |
| Morphology micro precision | 89.01% | **89.26%** | 83.32% | **83.65%** |
| Morphology micro recall | **96.72%** | 96.61% | **95.13%** | 95.03% |
| Morphology micro F1 | 92.70% | **92.79%** | 88.84% | **88.98%** |
| Morphology macro F1 | 91.67% | **91.81%** | 86.18% | **86.19%** |
| Morphology macro Average Precision | **94.07%** | 94.04% | **90.44%** | 90.41% |

The gain is small but controlled: the selected student improves development
loss, lemma accuracy, morphology precision, micro F1, and macro F1 across both
written standards. Bokmål also improves UPOS; Nynorsk UPOS and morphology
ranking quality remain effectively flat. The temperature ablation shows that
one global temperature of 2.0 over-softens the binary morphology outputs and
the 1,059-way lemma-rule distribution. Future work should therefore use
task-specific distillation policies rather than further blind global-weight
sweeps. Both official test splits remain untouched.

## NBdigital silver-source preparation

This is a data-provenance milestone, not a quality benchmark. The first
silver source is Språkbanken resource `oai:nb.no:sbr-43`, downloaded from its
official CC0 archive. Preparation uses an OCR-confidence floor of `0.95`, a
128-token sentence ceiling, normalized token-sequence deduplication, and
overlap exclusion against train, Development, and test for both Norwegian
written-standard profiles.

The official archive preparation completed in approximately 30 minutes:

- source archive SHA-256:
  `9d9c48843d4c9ac845ce775d98118bad667452abe259462770f4a975f23ed505`;
- retained documents: 936;
- retained sentences: 2,542,722;
- retained tokens: 50,385,644;
- generated JSONL size: approximately 880 MB.

Existing Oslo-Bergen task labels are not used. No Student result may be
attributed to silver data until offline Teacher pseudo-labeling, a fixed and
deterministic subset plus Gold/Silver mixing policy, and a controlled
comparison against the selected Student have all completed. The official test
splits remain untouched.

## Frozen morphology-head probe

This is a Development-only architecture diagnostic, not a released-model
benchmark. The selected 12-epoch Student is completely frozen at the
`morphology-pre-head` representation boundary. Seed 42 trains only matched
linear, shared residual MLP, and feature-specific residual MLP heads for eight
epochs. All controls use the checkpointed morphology weights and correction
strength `1.0`.

- Source checkpoint:
  `runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-w010-e12-weighted/best.pt`
- Report:
  `runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-w010-e12-weighted/morphology-head-probe-seed42.json`
- Training representations: 489,216 tokens
- Runtime including one-time extraction: 3 minutes 14 seconds
- Reusable FP32 cache: 439 MB

### Gender accuracy

| Standard and slice | Linear | Shared MLP | Feature MLP | Feature MLP − linear |
| --- | ---: | ---: | ---: | ---: |
| Bokmål overall | 94.0334% | 96.6262% | **97.1212%** | **+3.0878 pp** |
| Bokmål annotated | 82.1727% | 90.2025% | **91.8369%** | **+9.6642 pp** |
| Bokmål Rare | 83.6825% | 89.7778% | **90.7302%** | **+7.0476 pp** |
| Bokmål OOV | 84.8949% | **89.8112%** | 89.5262% | **+4.6313 pp** |
| Nynorsk overall | 92.8800% | 94.6752% | **95.1392%** | **+2.2592 pp** |
| Nynorsk annotated | 79.5429% | 84.6826% | **86.2062%** | **+6.6633 pp** |
| Nynorsk Rare | 81.8638% | 86.7112% | **87.9649%** | **+6.1011 pp** |
| Nynorsk OOV | 82.8470% | **86.5142%** | **86.5142%** | **+3.6672 pp** |

The feature-specific MLP also improves overall `Definite`, `Number`,
`PronType`, `VerbForm`, and `Degree` on both written standards. The only
overall regression against the linear probe is three Nynorsk `NumType`
decisions. Probe parameter counts are 10,036, 158,068, and 678,772
respectively.

The result identifies a genuine nonlinear head-capacity signal, but the
standalone probe does not execute the checkpoint's structured morphology
decoder, Bundle-32 reranker, or agreement path. The complete selected model is
still 0.2062/0.2592 overall `Gender` points and 1.5319/1.1830 OOV points above
the standalone feature MLP on Bokmål/Nynorsk. No production architecture is
selected from this one seed, and both official test splits remain untouched.

### Seed-42 convergence extension

The same cached representations and seed were rerun for 16 epochs. All
training losses continued to decrease, but Development metrics had largely
saturated:

| Gender comparison at epoch 16 | Bokmål | Nynorsk |
| --- | ---: | ---: |
| Shared MLP − linear, overall | +2.7276 pp | +2.1504 pp |
| Feature MLP − linear, overall | **+2.8981 pp** | **+2.1984 pp** |
| Feature MLP − shared MLP, overall | +0.1705 pp | +0.0480 pp |
| Feature MLP − shared MLP, annotated | +0.6040 pp | +0.0508 pp |
| Feature MLP − shared MLP, Rare | +0.5397 pp | +0.6686 pp |
| Feature MLP − shared MLP, OOV | **−0.5344 pp** | **+0.5521 pp** |

Relative to its own epoch-8 result, feature MLP improves overall Gender by
only 0.1870 Bokmål and 0.0416 Nynorsk points. Bokmål OOV loses three correct
tokens while Nynorsk OOV gains eight. Across every feature, feature MLP has
136 fewer Bokmål overall errors than shared MLP but 12 more Bokmål OOV errors;
for Nynorsk it has only nine fewer overall errors and 15 fewer OOV errors.

The extra 520,704 feature-MLP parameters therefore have a small and
slice-dependent advantage over shared MLP. This run confirms the nonlinear
head-capacity signal but does not select the larger head. The next cheap gate
is a matched 16-epoch second seed, not additional same-seed epochs or a full
Student run.

### Seed-43 replication and candidate selection

The matched second seed reused the same frozen representations and ran all
three probes for 16 epochs in 3 minutes 15 seconds:

- Report:
  `runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-w010-e12-weighted/morphology-head-probe-seed43-e16.json`
- Random seed: 43
- Epochs: 16
- Final losses: linear `0.014339`, shared MLP `0.006764`, feature MLP
  `0.005204`

| Gender comparison at epoch 16 | Bokmål | Nynorsk |
| --- | ---: | ---: |
| Shared MLP − linear, overall | +2.8486 pp | +2.1376 pp |
| Feature MLP − linear, overall | **+2.9366 pp** | **+2.3488 pp** |
| Feature MLP − shared MLP, overall | +0.0880 pp | +0.2112 pp |
| Feature MLP − shared MLP, annotated | +0.3375 pp | +0.7314 pp |
| Feature MLP − shared MLP, Rare | +0.0635 pp | +1.2537 pp |
| Feature MLP − shared MLP, OOV | **−0.3207 pp** | **−0.4338 pp** |

The OOV regressions equal nine additional Bokmål and eleven additional
Nynorsk Gender errors. The same trade-off is visible across the complete
feature inventory:

| Seed-43 errors | Shared MLP | Feature MLP | Feature MLP difference |
| --- | ---: | ---: | ---: |
| Bokmål overall | 2,223 | **2,167** | −56 |
| Bokmål Rare | 526 | **520** | −6 |
| Bokmål OOV | **599** | 610 | +11 |
| Nynorsk overall | 2,535 | **2,457** | −78 |
| Nynorsk Rare | 428 | **397** | −31 |
| Nynorsk OOV | **584** | 602 | +18 |

Across both 16-epoch seeds and both written standards, feature MLP removes
279 overall and 46 Rare feature errors relative to shared MLP, but adds 26
OOV errors. The result robustly confirms that a nonlinear morphology
transformation is warranted. It rejects the extra 520,704 feature-specific
parameters as the first production candidate because their small aggregate
gain does not generalize to unseen word forms.

The selected next end-to-end ablation is therefore one shared post-fusion
residual morphology MLP at `morphology-pre-head`. It remains schema- and
language-independent and feeds the existing independent heads, structured
decoder, Bundle-32 reranker, and agreement refinement. This diagnostic does
not change the selected production checkpoint or any official test result.

### Implemented full-training ablation

The probe-selected candidate is implemented behind the independent
`--morphology-pre-head-architecture` switch. `identity` is the unchanged
control and compatibility default; `shared-mlp` adds 148,032 parameters at
hidden size 192 after character fusion and before every morphology head. The
setting is checkpointed and automatically restored for evaluation,
distillation-teacher loading, and later probes. Focused tests cover
same-seed initialization parity for all common parameters, the
`morphology`-scoped direct-bundle gradient boundary, old-checkpoint fallback,
CLI parsing, strict state loading, and strict export parity.

The fixed full-training command is:

```shell
.venv/bin/python -m prism.languages.norwegian.train_baseline \
  --language-tag no \
  --model-role student \
  --checkpoint runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-w010-e12-weighted/best.pt \
  --teacher-checkpoint runs/no-teacher-base-character-cnn-e12-weighted/best.pt \
  --categorical-distillation-objective dkd \
  --distillation-temperature 1.0 \
  --distillation-weight 0.1 \
  --morphology-pre-head-architecture shared-mlp \
  --morphology-bundle-candidate-count 32 \
  --morphology-bundle-loss-weight 0.1 \
  --morphology-bundle-loss-gradient-scope morphology \
  --early-stopping-patience 4 \
  --epoch-count 12 \
  --morphology-weight-cap 10.0
```

This changes exactly one inference-time architectural variable relative to
the selected Student. The current checkpoint remains the control. No
candidate result or production selection exists until the loss-selected new
checkpoint has been evaluated separately on canonical Bokmål and Nynorsk,
including UPOS, complete UFeats, Lemmas, every morphology feature, and
Rare/OOV slices.

### Completed shared morphology-MLP training gate

The fixed twelve-epoch run completed in approximately 1 hour 54 minutes 8
seconds and selected the final epoch. Its 70,661,786-byte checkpoint is
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-w010-e12-weighted/best.pt`.
Every epoch produced a new lowest combined Development loss, so the run ended
cleanly at the fixed schedule limit rather than through early stopping.

| Joint Development signal | Selected identity control | Shared morphology MLP |
| --- | ---: | ---: |
| Combined loss | **0.112124** | 0.114135 |
| Bundle loss | 0.126770 | **0.118622** |
| Bundle candidate coverage | not used for selection | 98.2180% |
| UPOS accuracy | **98.8672%** | 98.8302% |
| Lemma-rule accuracy | **98.8648%** | 98.8367% |

The additional morphology capacity reduces bundle loss by 0.008148, or 6.43%
relative, but raises combined loss by 0.002011 and loses 0.0370/0.0281
percentage points of UPOS/Lemma-rule accuracy. This is the intended
morphology-specific signal accompanied by a small all-task risk. It neither
selects nor rejects the candidate: separate canonical Bokmål and Nynorsk
evaluation with the selected full logit correction, complete UFeats, every
feature, and Rare/OOV slices remains mandatory. The official test splits
remain untouched.

### Shared morphology-MLP Bokmål gate

The canonical Bokmål Development evaluation with the selected full
checkpoint-derived logit correction is complete:

| Bokmål Development metric | Selected identity control | Shared morphology MLP | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | **98.9991%** | 98.9689% | -0.0302 pp / -11 correct |
| UFeats F1 | 96.1588% | **96.6372%** | +0.4784 pp / +174 correct |
| Lemmas F1 | **98.9634%** | 98.9304% | -0.0330 pp / -12 correct |
| Rare morphology micro F1 | 95.5501% | **96.3839%** | +0.8338 pp |
| OOV morphology micro F1 | **94.5367%** | 94.3785% | -0.1582 pp |
| Rare lemma end-to-end | **97.4286%** | 97.3968% | -0.0318 pp |
| OOV lemma end-to-end | **93.3025%** | 92.8750% | -0.4275 pp |

The intended target feature improves strongly in-distribution: Gender overall
gains 0.4124 points, annotated Gender 1.5012 points, and Rare Gender 1.8095
points. OOV Gender instead regresses by 0.3919 points. Number improves on all
four slices, while most other feature changes are much smaller.

Against UDPipe 2.17, the candidate remains 0.0192 points ahead on UPOS but
trails by 1.4326 points on complete UFeats and 0.0440 points on Lemmas. Gender
remains the dominant external gap: 1.0366 points overall, 3.1977 points on
annotated tokens, 3.3968 points on Rare tokens, and 2.2800 points on OOV
tokens.

This is a substantial real UFeats gain, not a benchmark-only annotation
conversion. It nevertheless fails the standalone Bokmål OOV and all-task
no-regression ideal. The candidate remains undecided until the matched
canonical Nynorsk evaluation establishes whether the trade-off is robust
across both written standards. The official test splits remain untouched.

### Shared morphology-MLP Nynorsk gate and joint decision

The matched canonical Nynorsk Development evaluation completes the
predeclared gate:

| Nynorsk Development metric | Selected identity control | Shared morphology MLP | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | **98.7136%** | 98.6688% | -0.0448 pp / -14 correct |
| UFeats F1 | 93.9104% | **94.0768%** | +0.1664 pp / +52 correct |
| Lemmas F1 | **98.5856%** | 98.5568% | -0.0288 pp / -9 correct |
| Rare morphology micro F1 | 91.2468% | **91.9361%** | +0.6893 pp |
| OOV morphology micro F1 | 88.5645% | **88.9226%** | +0.3581 pp |
| Rare lemma end-to-end | **97.1166%** | 96.9494% | -0.1672 pp |
| OOV lemma end-to-end | **91.9164%** | 91.8375% | -0.0789 pp |

Gender improves by 0.1728 points overall, 0.5587 on annotated tokens, 0.3343
on Rare tokens, and 0.1972 on OOV tokens. Definite and Number also improve
overall, Rare, and OOV. Against UDPipe 2.17, the candidate remains 0.0960
points ahead on UPOS and trails by 1.5840 UFeats points and 0.2720 Lemmas
points.

Across both written standards, the shared MLP corrects 226 additional complete
UFeats bundles and removes 279 individual feature errors. It removes 204
Gender errors overall and 65 on Rare forms. The cost is 25 UPOS and 21 Lemma
predictions. On the combined 5,343 OOV tokens, it loses four exact UFeats
bundles, adds ten individual feature errors, and adds six Gender errors; the
Bokmål OOV regression is partly but not completely offset by Nynorsk.

This is accepted as the next compact architecture reference. The primary
remaining external deficit is complete UFeats/Gender, the gain appears on
both written standards and strongly on Rare forms, and the OOV regression is
small relative to 226 complete-split UFeats and 204 Gender corrections. The
OOV and lemma trade-offs remain explicit quality risks for the next
intervention; they must not be hidden by the aggregate selection.

`shared-mlp` is now the Norwegian CLI default for new training runs. The
generic model-building default and checkpoint fallback remain `identity`, so
old artifacts and other language integrations do not silently change.
`--morphology-pre-head-architecture identity` remains the explicit
reproduction control. Evaluation and export always use checkpoint metadata.
Official test splits remain untouched.

The resulting current canonical comparison is:

| Development metric | Prism Bokmål | UDPipe Bokmål | Difference | Prism Nynorsk | UDPipe Nynorsk | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UPOS F1 | 98.9689% | 98.9497% | +0.0192 pp | 98.6688% | 98.5728% | +0.0960 pp |
| UFeats F1 | 96.6372% | 98.0698% | -1.4326 pp | 94.0768% | 95.6608% | -1.5840 pp |
| Lemmas F1 | 98.9304% | 98.9744% | -0.0440 pp | 98.5568% | 98.8288% | -0.2720 pp |

These values use official-compatible gold-token Development scoring, the
canonical morphology policy, and logit-correction strength `1.0`. The
Nynorsk treebank policy has not yet been rerun on this checkpoint and is not
estimated from the previous model.

### Compositional bundle-scorer Bokmål gate

The controlled nonlinear scorer run is:

`runs/no-student-compositional-bundle-scorer-e12-weighted/best.pt`

It changes only the bundle residual scorer from `linear` to
`compositional-mlp`; the selected shared morphology MLP, direct bundle-loss
weight and gradient scope, data, seed, schedule, and canonical evaluation
policy remain matched. The checkpoint selected epoch 12.

| Bokmål Development metric | Selected linear scorer | Compositional scorer | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | 98.9689% | **99.0019%** | +0.0330 pp / +12 correct |
| UFeats F1 | **96.6372%** | 96.5960% | -0.0412 pp / -15 correct |
| Lemmas F1 | **98.9304%** | 98.8644% | -0.0660 pp / -24 correct |
| Rare UFeats F1 | **90.9841%** | 90.6032% | -0.3809 pp |
| OOV UFeats F1 | **86.7474%** | 86.2487% | -0.4987 pp |
| OOV Lemmas F1 | **92.8750%** | 92.2693% | -0.6057 pp |

The candidate fails the first canonical gate and is not selected. A Nynorsk
selection run is therefore not required to reject it as the new default.
`linear` remains the selected scorer and the selected production checkpoint
does not change.

The paired read-only task-interaction audits nevertheless identify useful
internal signal:

| Bokmål bundle audit | Selected linear scorer | Compositional scorer | Change |
| --- | ---: | ---: | ---: |
| Final errors | **1,223** | 1,238 | +15 |
| Missing-candidate errors | **72** | 90 | +18 |
| Ranking errors | 930 | **871** | -59 |
| Refinement errors | **221** | 277 | +56 |
| Gold candidate Top-1 on covered tokens | 97.3388% | **97.4773%** | +0.1385 pp |
| Hard candidate Top-1 over all tokens | 96.6510% | **96.7885%** | +0.1375 pp |

The scorer therefore learns a better top candidate, but the current static
residual marginalization gives almost the entire gain back. Hard candidate
decoding is only a diagnostic: it would lose the independent path's
open-combination behavior and is not selected as production decoding.
Relative to the selected final output, its 96.7885% value exposes about
0.1513 UFeats points of directly observable Bokmål headroom, not a promised
model improvement.

This result changes the next experiment. The scorer is not promoted, and
capacity is not increased again. The next gate is a frozen adaptive
probability-fusion probe. Its planning expectation is approximately
0.15--0.35 Bokmål UFeats points, with values above 0.4 considered optimistic.
This estimate is not a benchmark result and the gains of later stages are not
assumed to be additive.

### Planned path after the scorer rejection

The predeclared order is:

1. **Frozen adaptive fusion probe.** Freeze the selected checkpoint and train
   only a compact token- and feature-dependent gate on the training split.
   It mixes independent feature probabilities with marginalized bundle
   probabilities. Development and UDPipe outputs never train the gate.
2. **Candidate-coverage decomposition.** Without changing predictions, split
   missing gold bundles into training-seen-but-Top-32-pruned and never-seen
   combinations. Report coverage and rank curves for Top-32, Top-64,
   Top-128, and the complete training inventory.
3. **Open structured candidate generation.** If genuinely unseen bundles are
   material, generate a bounded beam from high-probability per-feature values
   and score combinations with a schema-derived energy function. This must
   retain independent feature confidences and a language-independent export
   contract.
4. **Final-output bundle objective.** Apply exact-bundle supervision or a
   consistency objective to the probabilities actually emitted after fusion,
   while retaining all per-feature losses. This addresses refinement errors
   instead of supervising only the pre-fusion candidate distribution.
5. **Lemma near-miss probe.** Only after morphology stabilizes, train a small
   frozen reranker over the audited top lemma rules. Add soft UPOS/morphology
   context only if it resolves errors beyond character and edit-rule evidence.
6. **Architecture-matched Teacher and silver data.** Retrain the Base Teacher
   with the final accepted output contract, label only under a fixed
   confidence policy, and compare the distilled Student with a matched
   gold-only Student. NorBERT4-large remains deferred until a demonstrated
   capacity limit.
7. **Final external gate.** Compare fixed Bokmål and Nynorsk Development
   outputs with UDPipe using UPOS, exact UFeats, Lemmas, all features, and
   Rare/OOV slices. Use the untouched test splits only once the complete
   Development policy is frozen.

Cleanup of rejected experiment paths is performed separately from these
model changes. The agreement refiner and rejected task-adapter paths can be
removed; diagnostic audits and the selected reproduction controls remain.
The compositional scorer stays only until the adaptive-fusion probe determines
whether its improved rank signal is useful.
