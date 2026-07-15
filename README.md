<p align="center">
  <img src="logos/logo.svg" alt="Prism" width="420">
</p>

# Prism

Prism is an experimental, modular NLP toolkit for fast, local, and
privacy-friendly linguistic analysis. The long-term goal is to provide
language-specific models through a unified API for tokenization, sentence
segmentation, part-of-speech tagging, lemmatization, and morphological
analysis across Python and Swift platforms.

The project currently focuses exclusively on **Norwegian Bokmål** and is being
built incrementally from small statistical baselines toward trainable neural
models.

## Current status

The current implementation can:

- read Universal Dependencies data in CoNLL-U format;
- build deterministic word, character, and POS vocabularies;
- train a word-only BiLSTM POS tagger;
- train a word-and-character BiLSTM POS tagger;
- analyze the morphological feature inventory in the treebank;
- jointly train POS tagging and morphological Number prediction through a
  shared word-and-character encoder;
- evaluate saved checkpoints on the official development and test splits;
- report accuracy by vocabulary status and precision, recall, and F1 by class;
- predict POS tags and Number values for externally tokenized text;
- train on Apple Silicon through PyTorch MPS, with a CPU fallback.

The best current POS-only model combines word and character representations
and reaches **96.00% UPOS accuracy** on the Norwegian Bokmål test split with
gold tokenization. The first multi-task model reaches **95.96% UPOS accuracy**
while additionally predicting morphological Number with **97.27% overall
accuracy** and **95.69% accuracy on tokens annotated for Number**. See
[the benchmark notes](docs/benchmarks.md) for the full setup and comparison.

Prism does **not** yet provide its own tokenizer, sentence segmenter,
lemmatizer, complete morphological analysis, dependency parser, or Swift
runtime. Morphological prediction currently covers only `Number`. Prediction
commands expect text that has already been split into tokens.

## Model architecture

The current neural models combine two representations for every token:

```text
Word ID -> word embedding -------------------+
                                              +-> sentence BiLSTM -+-> POS logits
Characters -> character embeddings -> BiLSTM +                    +-> Number logits
```

The word branch learns lexical information for frequent words. Words seen
fewer than twice during training are mapped to `<UNK>`. The character branch
still sees their spelling, allowing it to learn useful patterns such as
capitalization and Norwegian inflectional endings. A bidirectional sentence
LSTM then uses both left and right context. The POS-only model predicts one of
the 17 Universal POS tags for each token. The multi-task model reuses the same
contextual representation for separate POS and Number output layers, allowing
both training objectives to shape the shared encoder.

Three model variants are retained for comparison:

| Model | POS development | POS test | Number test |
| --- | ---: | ---: | ---: |
| Word-only BiLSTM | 91.91% | 91.22% | - |
| Word + character BiLSTM | 96.76% | 96.00% | - |
| POS + Number multi-task BiLSTM | 96.49% | 95.96% | 97.27% |

The Number score in this table includes the `<NONE>` class. Accuracy on test
tokens that carry a Number annotation is 95.69%. The rare combined value
`Plur,Sing` is retained by the current prototype but is not learned reliably
from its six training examples.

## Requirements

- macOS or another Python-compatible platform
- Python 3.12
- Git

Python 3.12 is used deliberately because it has broad compatibility with the
current PyTorch ecosystem. Python 3.14 is not currently the supported project
runtime.

## Setup

Clone the repository and enter it:

```bash
git clone git@github.com:dmlux/Prism.git
cd Prism
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install Prism in editable mode with its development dependencies:

```bash
python -m pip install -e './python[dev]'
```

The Python distribution is named `prism-nlp` because the name `prism` is
already occupied on PyPI. Its import package remains `prism`. Editable
installation ensures that source changes are used immediately without
reinstalling the package.

## Training data

The current experiments use the official
[UD Norwegian Bokmål](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal)
treebank. Download the dataset into the ignored local data directory:

```bash
git clone https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal.git \
  data/raw/UD_Norwegian-Bokmaal
git -C data/raw/UD_Norwegian-Bokmaal checkout \
  396d11f0c2bd290a2a2711015c04ac25bc3dcc06
```

The pinned commit is the version used for the documented benchmarks. The
dataset contains the original training, development, and test splits with
tokens, lemmas, UPOS tags, morphological features, and dependency annotations.

Training data is intentionally not committed to this repository.

## Training

Train the word-only baseline neural model:

```bash
python -m prism.baselines.recurrent.cli.train_pos
```

Train the word-and-character model:

```bash
python -m prism.baselines.recurrent.cli.train_character_pos
```

Train the joint POS and Number model:

```bash
python -m prism.baselines.recurrent.cli.train_pos_number
```

All commands automatically use Apple MPS when available and otherwise fall
back to the CPU. The word-only model trains for five epochs. The character and
multi-task experiments train for up to ten epochs and evaluate against the
development split after every epoch. POS-only checkpoints are selected by
development accuracy; the multi-task checkpoint is selected by the combined
POS and Number development loss.

Generated checkpoints are stored locally under:

```text
models/norwegian-bokmaal/
├── pos_bilstm.pt
├── pos_character_bilstm_10_epochs.pt
└── pos_number_bilstm.pt
```

The `models/` directory and common model formats are ignored by Git. Model
artifacts should be released separately with their training-data provenance,
configuration, benchmark results, and applicable license information.

## Evaluation

Evaluate the word-only model on the development split:

```bash
python -m prism.baselines.recurrent.cli.evaluate_pos
```

Evaluate the character model on the development split:

```bash
python -m prism.baselines.recurrent.cli.evaluate_character_pos \
  --checkpoint models/norwegian-bokmaal/pos_character_bilstm_10_epochs.pt
```

Evaluate the joint POS and Number model:

```bash
python -m prism.baselines.recurrent.cli.evaluate_pos_number
```

The evaluation commands accept `--split test` for a final evaluation of a
fixed model:

```bash
python -m prism.baselines.recurrent.cli.evaluate_pos_number --split test
```

Use the development split for model selection and tuning. The test split
should only be used to report results for an already fixed model version.

## Prediction

The prediction interface currently expects pre-tokenized input. Run the
word-and-character model with one shell argument per token:

```bash
python -m prism.baselines.recurrent.cli.predict_character_pos \
  --checkpoint models/norwegian-bokmaal/pos_character_bilstm_10_epochs.pt \
  Jeg leser en bok .
```

Example output:

```text
Jeg     PRON
leser   VERB
en      DET
bok     NOUN
.       PUNCT
```

The word-only comparison model remains available through:

```bash
python -m prism.baselines.recurrent.cli.predict_pos Jeg leser en bok .
```

Run the multi-task model to predict POS and Number together:

```bash
python -m prism.baselines.recurrent.cli.predict_pos_number \
  Jeg leser en bok og to bøker .
```

Example output:

```text
Token   POS     Number
Jeg     PRON    Sing
leser   VERB    <NONE>
en      DET     Sing
bok     NOUN    Sing
og      CCONJ   <NONE>
to      NUM     Plur
bøker   NOUN    Plur
.       PUNCT   <NONE>
```

In the future, Prism is intended to offer both a high-level raw-text API and a
lower-level API for applications, such as LexKeep, that already own their
tokenization and source offsets.

## Tests

Run the complete test suite from the repository root:

```bash
python -m pytest python/tests
```

The current tests cover CoNLL-U parsing, word, character, and morphological
feature encoding, two-level batch padding, POS and multi-task model tensor
shapes, and multi-task evaluation.

## Repository layout

```text
Prism/
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── benchmarks.md
│   └── model-strategy.md
├── logos/
├── python/
│   ├── pyproject.toml
│   ├── src/prism/
│   │   ├── baselines/
│   │   │   ├── dictionary.py
│   │   │   └── recurrent/
│   │   │       ├── cli/
│   │   │       ├── dataset.py
│   │   │       ├── inference.py
│   │   │       ├── model.py
│   │   │       ├── training.py
│   │   │       └── vocabulary.py
│   │   ├── analyze_morphology.py
│   │   ├── conllu.py
│   │   └── ...
│   └── tests/
└── README.md
```

Local datasets, checkpoints, virtual environments, caches, and generated
training artifacts are excluded through `.gitignore`.

## Roadmap

Planned milestones include:

- a versioned output schema for all supported morphology features and learned
  lemma edit rules;
- a high-capacity pretrained Norwegian teacher and a compact distilled student;
- calibrated confidence scores and explicit handling of uncertain predictions;
- versioned ExecuTorch model artifacts with PyTorch export-parity tests;
- a native Swift runtime wrapper and document-scale latency benchmarks;
- language-aware tokenization and sentence segmentation as a separate task;
- modular Swift packages such as `PrismKit`, `PrismCore`, and
  `PrismNorwegian`;
- additional language modules after the Norwegian pipeline is stable.

Phrase, named-entity, and multiword-expression recognition are considered
separate span-level tasks and are not part of the current POS model.

## Licensing

The Prism source code is licensed under the
[Apache License 2.0](LICENSE.md). This software license applies to Prism's own
source code and does not relicense external datasets or model artifacts.

The UD Norwegian Bokmål data used by the current experiments is distributed
under **CC BY-SA 4.0**. Its attribution and share-alike requirements must be
reviewed and documented when redistributing trained model artifacts. Dataset
files are not included in this repository. Model releases must carry their own
provenance, attribution, and applicable license information.

## Project direction

Prism aims to become a modular, open-source NLP toolkit for tokenization,
sentence segmentation, POS tagging, lemmatization, and morphological analysis.
It is designed around local inference, explicit language modules, reproducible
training, and a unified API that can eventually be consumed from both Python
and Swift applications.
