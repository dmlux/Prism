# Prism Benchmarks

## Norwegian Bokmål POS tagging

All results use the official gold tokenization and the original
train, development, and test splits.

### Dataset

- Source: UniversalDependencies/UD_Norwegian-Bokmaal
- Commit: `396d11f0c2bd290a2a2711015c04ac25bc3dcc06`
- License: CC BY-SA 4.0
- Training sentences: 15,696
- Development sentences: 2,409
- Test sentences: 1,939

### Shared training configuration

- Python: 3.12.13
- PyTorch: 2.13.0
- Device: Apple MPS
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 5 for the initial POS experiments; up to 10 for the
  extended character and multi-task experiments
- Training batch size: 32
- Random seed: 42
- Minimum word frequency: 2
- Word vocabulary size: 12,390
- POS classes: 17

### Word-only BiLSTM

- Word embedding size: 64
- Sentence BiLSTM hidden size: 128 per direction
- Development accuracy: 91.91%
- Test accuracy: 91.22%

### Word and character BiLSTM

- Word embedding size: 64
- Character embedding size: 32
- Character BiLSTM hidden size: 32 per direction
- Sentence BiLSTM hidden size: 128 per direction
- Character vocabulary size: 115
- Development accuracy: 96.43%
- Test accuracy: 95.75%

### Extended word and character BiLSTM

The same architecture was trained for up to 10 epochs while retaining
the checkpoint with the best development accuracy. The selected checkpoint
was saved after epoch 9.

- Development accuracy: 96.76%
- Test accuracy: 96.00%
- Test accuracy on known tokens: 97.18%
- Test accuracy on `<UNK>` tokens: 88.87%

Adding character representations reduced the test error rate
from 8.78% to 4.25%, a relative reduction of approximately 52%.

The test split must not be used for model selection or
hyperparameter tuning.

### Accuracy by word-vocabulary status

The test split contains 25,706 known tokens and 4,260 tokens
mapped to `<UNK>`.

| Model | Known tokens | `<UNK>` tokens |
| --- | ---: | ---: |
| Word-only BiLSTM | 95.25% | 66.85% |
| Word and character BiLSTM | 96.87% | 88.94% |

Character representations improved accuracy on `<UNK>` tokens
by 22.09 percentage points and reduced their error rate by
approximately 67%.

### POS and Number multi-task BiLSTM

The multi-task model shares its word, character, and sentence
encoder between POS tagging and Number prediction. Separate
output layers predict the two tasks.

The checkpoint was selected by the lowest combined development
loss and was saved after epoch 6.

| Metric | Development | Test |
| --- | ---: | ---: |
| POS accuracy | 96.49% | 95.96% |
| Number accuracy | 97.50% | 97.27% |
| Number accuracy on annotated tokens | 95.78% | 95.69% |

Number results by value on the test split:

| Value | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `<NONE>` | 98.76% | 98.19% | 98.47% | 18,869 |
| `Plur` | 94.31% | 94.31% | 94.31% | 3,145 |
| `Plur,Sing` | 0.00% | 0.00% | 0.00% | 2 |
| `Sing` | 94.95% | 96.26% | 95.60% | 7,950 |

The multi-task model retains nearly the same POS accuracy as the
best POS-only model while additionally predicting Number.

## NorBERT4-xsmall student without distillation

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
