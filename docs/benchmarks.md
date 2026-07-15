# Vexo Benchmarks

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
- Epochs: 5
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

Adding character representations reduced the test error rate
from 8.78% to 4.25%, a relative reduction of approximately 52%.

The test split must not be used for model selection or
hyperparameter tuning.