# Prism Benchmarks

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

## Shared Norwegian NorBERT4-Base teacher

The first teacher uses the same supervised data, schema, task heads, and
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
