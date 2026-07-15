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
