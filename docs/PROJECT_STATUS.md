# Prism project status

Last updated: 2026-08-01

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
- External gold-token comparisons use the official UD word-level definitions
  for `UPOS`, complete universal `UFeats`, and `Lemmas`. The three values are
  precision/recall/F1 plus aligned accuracy; with fixed gold tokenization they
  are numerically equal, but counts and all four fields remain serialized.
- The same official-compatible metrics are accumulated separately for every
  configured token-frequency slice. Norwegian evaluation now prints and
  serializes Rare/OOV `UD UPOS F1`, exact complete-bundle `UD UFeats F1`, and
  `UD Lemmas F1` alongside the existing task-specific slice diagnostics.
- Rare/OOV evaluation also preserves the exact integer correct count for every
  morphology feature and prints a ranked feature-error attribution. Its
  `errors` column is the number of wrong feature decisions; `share` uses all
  per-feature errors in that slice as its denominator. One token can therefore
  contribute to more than one feature, which keeps this diagnostic distinct
  from exact complete-bundle `UD UFeats`.
- External feature comparison is opt-in through
  `--morphology-feature-comparison`. Omitting it performs standalone Prism
  evaluation and does not read or launch a comparison system.
- `XPOS`, `AllTags`, tokenization, and dependency-parser scores are not emitted
  because the current Prism bundle does not predict those outputs.

## UDPipe 2.17 comparison benchmark

The reproducible gold-token comparison path is implemented. Prism evaluation
now emits official-compatible `UPOS`, `UFeats`, and `Lemmas` metrics, restores
the Norwegian treebank's normalized `$` lemma marker before scoring, and
records the complete score objects in JSON. The separate
`prism.languages.norwegian.benchmark_udpipe` command obtains or reuses
versioned UDPipe CoNLL-U predictions and scores them with the same local
implementation. Its output was cross-checked against the official updated
CoNLL-2018 evaluator on the complete Bokmål development split with identical
results.

UD 2.17 is pinned separately at Bokmål commit
`b8618a2b935762d6ccd2dc997180c3e46f74f6b7` and Nynorsk commit
`2bbe9c67d5e81eadf237b7840ebac31bffca38ae`. SHA-256 checks confirm that all
six Norwegian train/development/test CoNLL-U files are byte-identical to the
currently pinned files. The selected DKD Student therefore already used the
same training and development content as the UDPipe 2.17 comparison; no
retraining is required for this development result. New comparison checkpoints
can nevertheless select `--treebank-release 2.17`, and store the release plus
both exact revisions.

The current selected shared-MLP Student, evaluated canonically with the fixed
full checkpoint-derived logit correction, compares as follows:

| Development metric | Prism Bokmål | UDPipe Bokmål | Difference | Prism Nynorsk | UDPipe Nynorsk | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UPOS F1 | 98.9689% | 98.9497% | +0.0192 pp | 98.6688% | 98.5728% | +0.0960 pp |
| UFeats F1 | 96.6372% | 98.0698% | -1.4326 pp | 94.0768% | 95.6608% | -1.5840 pp |
| Lemmas F1 | 98.9304% | 98.9744% | -0.0440 pp | 98.5568% | 98.8288% | -0.2720 pp |

Prism therefore leads UDPipe 2.17 slightly on UPOS for both written standards.
Complete-bundle UFeats remains the largest external quality gap; Lemmas is
near parity on Bokmål and has a larger Nynorsk deficit. These are canonical
model-quality values. The optional Nynorsk treebank output policy has not yet
been rerun for the selected shared-MLP checkpoint and is not inferred from an
older checkpoint. Full numbers and commands are recorded in
`docs/benchmarks.md`. The official test splits remain unevaluated.

### UFeats recovery plan

The gap is concentrated rather than model-wide. Against the selected DKD
Student, `Gender` trails UDPipe by 3.3297 points on Bokmål and 2.8800 points on
Nynorsk; the next largest gaps are `Number` and `Definite`. Nynorsk additionally
exposes incompatible treebank conventions: `Gender=Com` has substantial gold
support while singular `Number` and definite `Definite` are almost or entirely
absent, although the joint Norwegian model learns those values from Bokmål.

The accepted order is:

1. evaluate a reversible class-weight logit correction, using only the
   training-derived weights already stored in the checkpoint;
2. separate Prism's canonical Norwegian morphology from optional Bokmål-UD and
   Nynorsk-UD annotation/output policies without splitting the shared model;
3. evaluate a compact complete-bundle reranker after the first two effects are
   measured.

Step 1 is implemented and selected. The evaluation-only
`--morphology-logit-correction-strength` option subtracts the selected fraction
of `log(class_weight)` before decoding. Loss and model parameters remain
untouched, and the JSON analysis records the resolved strength and checkpoint
weight source. The complete predeclared Development grid
`0.0/0.25/0.5/0.75/1.0` selects the full correction at `1.0` as the shared
Norwegian output policy: UFeats rises from 93.9151% to 95.7601% on Bokmål and
from 91.5232% to 93.3920% on Nynorsk. It closes 44.4% and 45.2% of the
respective Development gap to UDPipe while preserving the strongest joint
tradeoff across exact UFeats, per-label, and Rare/OOV quality. Both test splits
remain untouched.

The CLI default remains zero for legacy and unweighted checkpoint
compatibility. A released artifact must not depend on that CLI default: it
stores the selected `1.0` policy together with the per-feature correction
offsets derived from the checkpointed training weights. Native runtimes apply
those fixed offsets before morphology decoding, and parity tests cover raw
logits, corrected logits, and decoded labels.

The export contract now implements this requirement. Both tensor-only token
tagger export adapters accept a typed correction, convert its resolved
`strength * log(weight)` vectors into registered model buffers, and subtract
them inside the strictly exported graph. The selected character-aware path has
eager-versus-`torch.export` parity coverage. A Swift or C++ runtime consuming
that artifact therefore receives corrected morphology logits and cannot omit
the correction accidentally. The future production artifact builder must
resolve the selected checkpoint policy and construct this corrected adapter;
manifest serialization and complete backend lowering remain open.

Step 2 now has a reversible evaluation implementation behind
`--ud-morphology-policy treebank`. It leaves Prism's canonical predictions,
per-label metrics, and Rare/OOV metrics unchanged and maps morphology only
before official complete-bundle UFeats scoring. The initial Norwegian policy
normalizes combined feminine/masculine adjective and determiner gender to
`Com`; the Nynorsk profile additionally omits singular `Number` and definite
`Definite`, matching the feature-expression policy observed in its pinned
training split. The canonical policy remains the backward-compatible default.
The ablation must run with the selected logit correction on both Development
splits before this policy can be selected or rejected.

Both Development runs are complete. Bokmål remains exactly unchanged at
95.7601% UFeats. Nynorsk rises from 93.3920% canonical UFeats to 95.9136% under
the treebank policy, a 2.5216-point gain, and exceeds the reproduced UDPipe
2.17 result of 95.6608% by 0.2528 points. Canonical per-label and Rare/OOV
metrics remain unchanged by construction. This demonstrates that the former
Nynorsk deficit was dominated by external annotation conventions rather than
missing contextual capacity. The Bokmål model-quality gap remains 2.3097
points.

The evaluator now implements that audit generically. Named UD feature-policy
steps run in fixed order and report the marginal number of changed, improved,
and regressed complete bundles relative to the immediately preceding step.
The counts are printed and serialized with the official UD metrics. The
Norwegian treebank policy is decomposed into `common-gender`,
`nynorsk-number`, and `nynorsk-definite`. The completed Bokmål audit changes
five already-wrong bundles and produces no improvement or regression. On
Nynorsk, the three steps improve 611, 134, and 43 complete bundles
respectively, with zero regressions. The 788 improvements exactly explain the
2.5216-point gain over 31,250 tokens. The external Nynorsk treebank policy is
therefore selected and frozen on Development; canonical mixed-Norwegian
output remains a separate contract.

Step 3 now has a training-only candidate-space oracle. The reusable evaluator
builds a frequency-ranked inventory of complete morphology bundles per UPOS
from the joint Bokmål/Nynorsk training splits and measures whether each
Development gold bundle is present globally, under its gold UPOS, and in the
top 1/2/4/8/16/32 candidates. Empty bundles are reported separately from
annotated tokens so `<NONE>` cannot inflate the decision. The inventory has
256 distinct bundles, 298 UPOS-bundle pairs, and at most 74 candidates for one
UPOS. It covers every but one annotated Bokmål token and 99.1861% of annotated
Nynorsk tokens under gold UPOS.

| Annotated gold-bundle oracle | Bokmål | Nynorsk |
| --- | ---: | ---: |
| Top 4 per gold UPOS | 68.2257% | 56.1072% |
| Top 8 per gold UPOS | 80.6622% | 74.9308% |
| Top 16 per gold UPOS | 93.0986% | 90.3684% |
| Top 32 per gold UPOS | 98.7736% | 94.8125% |
| Full training inventory | 99.9953% | 99.1861% |

This supports a compact bundle-aware scorer, but rejects an overly narrow
top-4 or top-8 candidate space. The first reranker ablation should score up to
32 training-derived candidates jointly with the independent-head logits and
must retain an independent-decoder fallback for unseen bundles and predicted
UPOS errors. These oracle figures are ceilings, not achieved model scores;
they use gold UPOS only to isolate candidate coverage. Both test splits remain
untouched.

The optional Top-32 reranker is implemented and selected as the new Norwegian
Student standard. `--morphology-bundle-candidate-count 32` builds its candidate
specification from the joint training targets and trains it jointly through
the existing morphology losses. It consumes soft predicted UPOS, independent
feature likelihoods, and a learned token-to-candidate residual; gold UPOS is
not an inference input. Per-feature learned gates add bundle evidence
residually, so the independent path remains capable of unseen combinations.
The complete specification is checkpointed and restored for evaluation and
export. `--disable-morphology-bundle-reranker` provides a matched checkpoint
diagnostic, and strict export parity is covered.

Against the separately trained previous DKD Student, canonical Development
UFeats rises from 95.7601% to 96.0076% on Bokmål and from 93.3920% to 93.6384%
on Nynorsk. UPOS also improves on both standards. Lemma accuracy improves by
0.0138 points on Bokmål and regresses by 0.0192 points on Nynorsk, while
Rare/OOV morphology improves on both standards and OOV lemma accuracy improves
on both. This broad real-output improvement selects the reranker; the external
UDPipe comparison is not the selection criterion. Disabling the residual path
inside the trained checkpoint causes a large drop, but is treated only as an
integration diagnostic because the independent heads co-adapted during joint
training.

With the already selected external treebank policy, Bundle-32 reaches 96.0048%
UFeats on Bokmål and 96.1920% on Nynorsk. The Bokmål policy changes six bundles,
improves none, and regresses one, so canonical output is marginally better and
the gap to UDPipe 2.17 remains 2.0650 points. The Nynorsk policy improves 798
bundles with zero regressions and places Prism 0.5312 points ahead of the
reproduced UDPipe result. These policy scores describe the external annotation
contract, not additional learned model quality. Both test splits remain
untouched.

The export/runtime decision is explicit: one shared Norwegian model graph
serves all three requested profiles. `nb` uses canonical morphology, `nn` uses
the Nynorsk treebank policy only when external Nynorsk-UD output is requested,
and mixed or unspecified `no` remains canonical. The Nynorsk transformation
must never be baked unconditionally into the shared model. A future manifest
stores the versioned output profiles and their defaults; Swift and C++ select
the requested profile after neural decoding. Separate `nb` and `nn` package
wrappers may share identical model bytes. Manifest serialization, native
profile selection, and cross-runtime parity tests for `nb`, `nn`, and `no`
remain open implementation tasks.

The resolved Norwegian Top-32 specification contains 185 candidates. At the
Student hidden size of 192, the component adds 35,723 trainable parameters,
approximately 143 KB of raw FP32 parameter values before export or
quantization.

A reusable token-aligned morphology error audit is implemented. Evaluation can
select a feature with `--morphology-error-audit-feature` and optionally compare
against an aligned CoNLL-U system prediction with
`--morphology-error-audit-comparison`. The audit records every error plus
aggregates by confusion, gold and predicted UPOS, training-frequency class,
normalized form, and neighboring gold-UPOS context. It separately counts
comparison-system feature correctness and complete-bundle correctness. The
observer consumes the exact decoded predictions from the normal evaluation
loop; it does not introduce a second inference path.

The first Bokmål Gender audit is complete. Bundle-32 makes 1,033 Gender errors
over 36,369 Development tokens. UDPipe gets the Gender feature right on 772 of
those errors and the complete bundle right on 756. Errors are distributed over
503 frequent, 261 rare, and 269 OOV tokens; 561 are NOUN, 191 ADJ, and 142
PROPN. The largest confusion is `Fem -> Masc` with 212 instances. This rejects
a frequency-only or static-lexicon-only explanation.

A checkpoint-candidate cross-check further shows that the gold bundle is
already present in the Top-32 inventory under predicted UPOS for 876 of the
1,033 Gender errors. Among the 756 errors whose complete bundle UDPipe solves,
672 already have that candidate available and only 84 do not. Increasing the
candidate limit can therefore address only a small minority of the measured
gap.

A bounded sentence-level morphology agreement refiner is now implemented as
an optional, unselected ablation. With
`--morphology-agreement-window-radius 3`, it attends only to valid neighboring
tokens within three positions, excludes the current token, and combines their
token representations, soft UPOS, and current morphology probabilities in a
64-dimensional bottleneck. Gated residual heads refine only `Definite`,
`Gender`, and `Number`; all other feature logits remain on the selected path.
The component adds 29,707 parameters (about 119 KB raw FP32), is checkpointed,
can be disabled during evaluation, and passes strict export coverage. It uses
no gold labels at inference.

The completed 12-epoch ablation selected epoch 10 and is rejected. Compared
with the selected Bundle-32 Student, Bokmål UFeats falls from 96.0076% to
95.7821%; annotated `Definite`, `Gender`, and `Number` fall by 0.1021, 0.3819,
and 0.1319 points. Rare and OOV morphology fall by 0.0778 and 0.2779 points.
Nynorsk UFeats improves from 93.6384% to 93.7824% and Rare morphology improves
by 0.3828 points, but OOV morphology falls by 0.1219 points. The component
therefore fails the required shared-standard and target-feature criteria.
Bundle-32 remains selected; both test splits remain untouched.

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

- `--task-head-architecture` selects the `linear` Format-3 control or
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

The intermediate `shared-mlp`, `wide-shared-mlp`, Task-Adapter, and
structured-without-character variants below are retained as selection history
only. Their executable enum, CLI, model, and test paths were removed in the
July 2026 cleanup.

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

The subsequent official-compatible Bokmål re-evaluation records 99.2824%
UPOS F1, 95.3532% complete-bundle UFeats F1, and 98.9057% Lemmas F1 for the
uncorrected canonical Teacher. Against UDPipe 2.17 this is +0.3327, -2.7166,
and -0.0687 percentage points respectively. The Teacher is therefore clearly
stronger on UPOS but not an across-the-board external winner. Its raw UFeats
also trails the later selected corrected Bundle-32 Student at 96.0076%,
because the Teacher predates both the bundle reranker and its selected output
correction. This is now a hard gate for silver labeling: uncorrected Teacher
morphology cannot be treated as pseudo-gold.

The predeclared full class-weight correction improves Teacher Bokmål UFeats
from 95.3532% to 96.3953%, Rare morphology micro F1 from 96.2034% to 96.5756%,
and OOV morphology micro F1 from 94.8188% to 95.6278%. UPOS and Lemmas are
unchanged. Corrected Teacher UFeats is 0.3877 points above the selected
Bundle-32 Student but remains 1.6745 points behind UDPipe; Lemmas remains
0.0687 points behind UDPipe. This accepts correction as necessary if the
existing Teacher is used for pseudo-labels, but does not yet accept that
Teacher as the final source. The corresponding corrected canonical Nynorsk
evaluation remains the next gate.

The corrected canonical Nynorsk gate is now complete: 98.8160% UPOS, 94.4896%
UFeats, and 98.5888% Lemmas. These exceed the selected canonical Bundle-32
Student by 0.1440, 0.8512, and 0.0544 percentage points. Rare/OOV morphology
micro F1 also lead by 0.6123/1.8500 points. Against UDPipe, UPOS leads by
0.2432 points while UFeats and Lemmas trail by 1.1712/0.2400 points. The
corrected existing Teacher is therefore a valid control and potentially useful
pseudo-labeler, but not the final choice: it predates the selected Bundle-32
reranker. A single architecture-matched Bundle-32 Teacher will be trained and
must beat this fixed control before any expensive silver-label run begins.

The Bundle-32 Base Teacher completed its meaningful learning curve before a
manual stop during epoch 6. Epoch 3 remains the saved loss-selected checkpoint;
epochs 4 and 5 both failed to improve, matching a patience-2 early-stopping
decision. The 609,716,974-byte checkpoint is
`runs/no-teacher-base-character-cnn-bundle32-e12-weighted/best.pt`.

With full correction, exact UFeats improves over the historical Teacher by
0.2282 points on Bokmål and 0.2400 points on Nynorsk; Rare morphology improves
by 0.0924/0.2412 points. Nynorsk UPOS and OOV morphology also improve.
Development loss and Lemmas regress slightly on both standards; Bokmål UPOS
and OOV morphology also regress. Bundle-32 is therefore the primary Base
morphology control, not a universal replacement. The historical corrected
Teacher remains an agreement control for later pseudo-label confidence. No
silver labels have been generated yet.

NorBERT4-large is deferred before any multi-hour run. The Base learning curve
shows that capacity alone is not the established bottleneck: the saved
epoch-3 checkpoint has the lowest summed Development loss, while lemma-rule
accuracy continues to improve through epochs 4 and 5. Moreover, official
UFeats scores complete feature bundles, but the current bundle reranker is
trained only indirectly through the individual feature losses; it has no
direct gold-bundle objective. The lemma edit-rule head also predicts its 1,059
rules independently from soft UPOS and morphology decisions. The next gate is
therefore a task-aligned error and ranking audit, followed by a measured
objective or decoder change on Base. Large becomes a candidate again only if
that corrected Base formulation still shows capacity-limited learning.

The first task-aligned correction is implemented but not yet benchmarked. A
typed, optional morphology-bundle loss now maximizes the total probability of
the complete gold bundle across every matching Top-32 candidate. It preserves
the independent per-feature objectives, masks gold bundles absent from the
training-derived inventory, and reports both its loss and covered-token ratio.
The legacy Bundle-32 control remains reproducible with loss weight `0`; the
first predeclared candidate uses weight `0.1`.

Training also supports automatic early stopping. The conservative default
patience is four complete epochs without lower combined Development loss;
zero disables it. Patience 2 is rejected as the default because earlier
Student runs produced a new best checkpoint at epoch 12 after epoch 9 had
remained best through two intervening epochs. The maximum epoch count and
learning-rate schedule remain explicit, and checkpoint selection still uses
combined Development loss. No claim of improvement exists until the new
70-MB Student is evaluated separately on canonical Bokmål and Nynorsk UPOS,
UFeats, Lemmas, per-feature, and Rare/OOV reports.

The fixed twelve-epoch Student run is now complete after approximately
1 hour 50 minutes 35 seconds. Epoch 12 is again the loss-selected checkpoint,
so the new objective still had useful learning signal at the configured
boundary. Bundle loss fell from 0.222202 at epoch 1 to 0.112506 at epoch 12;
candidate coverage remained constant at 98.2180%. The 70,068,398-byte
checkpoint is
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-w010-e12-weighted/best.pt`.
Its combined Development loss of 0.112011 includes the new weight-0.1
auxiliary term and is therefore not numerically comparable with the old
objective. Separate canonical Bokmål and Nynorsk evaluation remains required
before accepting the objective or extending its schedule.

Bokmål canonical evaluation with the selected full logit correction is
complete. Direct bundle supervision raises UFeats from the fixed Bundle-32
control's 96.0076% to 96.7390%, a substantial gain of 0.7314 percentage
points. The remaining gap to UDPipe 2.17 falls to 1.3308 points. Rare and OOV
morphology micro F1 improve by 0.7653 and 0.6938 points. UPOS changes by only
-0.0082 points versus the control and remains 0.0467 points above UDPipe.
However, Lemmas regresses by 0.0632 points and OOV lemma end-to-end accuracy
regresses materially by 0.7838 points; Rare lemma also loses 0.1270 points.
The candidate therefore proves that the aligned objective closes real UFeats
error, but it is not yet accepted as the all-task Student. Nynorsk evaluation
is required before deciding whether to rebalance or isolate the auxiliary
gradient.

Nynorsk confirms the morphology effect. Canonical UFeats rises from 93.6384%
to 94.4672%, a gain of 0.8288 points that closes 41.0% of the previous gap to
UDPipe 2.17. Rare and OOV morphology micro F1 improve by 1.0243 and 0.8011
points. UPOS improves by 0.0032 points and leads UDPipe by 0.1024 points.
Lemmas regresses by 0.0288 points, Rare lemma end-to-end by 0.0418 points, and
OOV lemma end-to-end by 0.0789 points.

Across both written standards, direct bundle supervision therefore produces a
large and consistent real-world morphology improvement, not a benchmark-only
policy effect. It also creates a repeatable lemma tradeoff, including a
material 0.7838-point Bokmål OOV-lemma regression. The candidate is not
selected as the all-task Student and its schedule is not extended. The next
controlled architecture change preserves the bundle objective while
preventing its auxiliary gradient from changing the shared representation and
independent task heads.

That gradient-isolation ablation was implemented. The explicit scope
`--morphology-bundle-loss-gradient-scope residual-only` gives the direct auxiliary loss
numerically identical candidate scores, but autograd can update only the
bundle reranker's token-to-candidate projection. The evidence derived from
UPOS and independent morphology logits, the input token representation,
Backbone, shared MLP, task heads, and refinement gates are detached for this
term. Normal supervised losses, distillation, the forward decoder, checkpoint
parameters, inference, and export remain unchanged. The resolved boolean is
stored under `training_config` in new checkpoints. The option requires a
positive bundle-loss weight and is disabled by default, preserving old runs.
Focused tests verify both score equality and the exact gradient boundary.

The fixed isolated-gradient training run is complete after approximately
1 hour 50 minutes 59 seconds. Epoch 12 again produced the lowest combined
Development loss. The 70,068,462-byte checkpoint is
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-isolated-w010-e12-weighted/best.pt`;
its metadata confirms isolation enabled, bundle weight `0.1`, and patience 4.
Compared with the unisolated direct-loss checkpoint, the joint Development
bundle loss is higher at 0.132241 versus 0.112506, as expected from the much
narrower gradient path. Lemma-rule loss improves from 0.046031 to 0.045161 and
lemma-rule accuracy from 98.7938% to 98.8618%; both also improve over the
Bundle-32 control's 0.045614 and 98.8426%. Morphology loss returns to 0.008676,
essentially the control's 0.008672 rather than the unisolated run's 0.008133.
This is promising evidence that isolation protects the shared tasks, but it is
not a model-selection result. Canonical Bokmål and Nynorsk UFeats, Lemmas,
UPOS, and Rare/OOV evaluation remain required. Because epoch 12 won again, a
longer schedule becomes eligible only after this candidate passes that
two-standard gate.

Bokmål canonical evaluation confirms that isolation protects the shared
tasks. Versus the selected Bundle-32 control, UFeats improves slightly from
96.0076% to 96.0351% and Lemmas from 98.9607% to 98.9881%. Rare morphology
micro F1 gains 0.1250 points and OOV morphology gains 0.0339 points; Rare and
OOV lemma end-to-end gain 0.2222 and 0.0356 points. Rare/OOV UPOS are
unchanged. Overall UPOS loses 0.0192 points, equivalent to seven additional
errors over 36,369 tokens, but remains 0.0357 points above UDPipe 2.17. Lemmas
now leads UDPipe by 0.0137 points.

Relative to the unisolated direct objective, isolation recovers 0.0907 Lemmas
points and 0.8194 OOV-lemma points, but gives back 0.7039 of its 0.7314 UFeats
gain. This establishes that the earlier lemma regression came from the broad
auxiliary gradient, while residual-only supervision is probably too narrow to
retain most of the bundle improvement. The balanced candidate remains viable
but unselected until the identical Nynorsk evaluation completes.

Nynorsk completes the gate and accepts the isolated candidate. Versus
Bundle-32, UFeats gains 0.1824 points, Lemmas gains 0.0096 points, Rare
morphology gains 0.0779 points, and OOV morphology and lemma gain 0.1100 and
0.2366 points. Overall UPOS loses 0.0096 points, Rare/OOV UPOS lose 0.0418 and
0.0394 points, and Rare lemma loses 0.1253 points. In counts, these are three
additional overall UPOS errors, one additional Rare-UPOS error, one additional
OOV-UPOS error, and three additional Rare-lemma errors over the fixed
Development split; they are not material relative to the consistent gains.

The isolated direct-bundle checkpoint is selected as the new compact
Norwegian reference. Across Bokmål and Nynorsk it improves canonical UFeats,
Lemmas, Rare/OOV morphology, and OOV lemma over Bundle-32. Its small UPOS and
Nynorsk Rare-lemma trades remain explicit. The unisolated checkpoint remains a
useful morphology upper control but is rejected as the all-task model. Because
the selected isolated run again chose epoch 12, the previously gated
30-epoch convergence run is now authorized; it must retain the same objective,
gradient scope, data, seed, and selection rule, with only the maximum schedule
length changing.

The 30-epoch convergence run is now complete after approximately 2 hours
26 minutes 51 seconds. Epoch 12 again has the lowest combined Development loss
at 0.117648; its UPOS accuracy is 98.8273%, lemma-rule accuracy 98.8293%,
bundle loss 0.132431, and bundle candidate coverage 98.2180%. Epochs 13
through 16 did not improve the selection loss, so patience 4 stopped the run
after epoch 16 and avoided the remaining 14 configured epochs. The longer
schedule therefore did not move selection past epoch 12. Because it also
changed the warmup/decay trajectory, this training signal alone neither
selects nor rejects the checkpoint. Separate canonical Bokmål and Nynorsk
evaluation against the fixed twelve-epoch isolated reference is still
required.

Canonical Bokmål evaluation rejects the longer schedule as a shared
all-task reference. Against the selected twelve-epoch isolated checkpoint,
UFeats improves from 96.0351% to 96.1643% and OOV morphology micro F1 from
94.2055% to 94.5146%. In exchange, UPOS falls by 0.0330 points, Lemmas by
0.0632, Rare morphology by 0.1791, Rare lemma by 0.3492, and OOV lemma by
0.1781 points. In complete-split counts, the candidate gains 47 exact UFeats
bundles but loses 12 UPOS and 23 Lemma predictions. It narrows the Bokmål
UFeats gap to UDPipe from 2.0347 to 1.9055 points, but gives up the previous
Lemma lead and almost the entire UPOS lead. Nynorsk evaluation is still
required to finish the convergence diagnosis; it cannot reverse the Bokmål
selection failure.

Canonical Nynorsk evaluation completes and rejects the convergence candidate.
Against the selected twelve-epoch isolated checkpoint, overall UPOS gains
0.0192 points, UFeats 0.0512, Lemmas 0.0096, and Rare lemma 0.0418. Rare UPOS
regresses by 0.1254 points, Rare morphology by 0.2972, OOV morphology by
0.0336, and OOV lemma by 0.1972; OOV UPOS is unchanged. The complete split
gains only 6 UPOS, 16 exact UFeats bundles, and 3 Lemma predictions. These
small Nynorsk gains cannot compensate for its frequency-slice regressions and
the failed Bokmål all-task gate.

The 30-epoch convergence ablation is therefore closed and rejected. The
twelve-epoch isolated direct-bundle checkpoint remains the selected compact
Norwegian reference. More epochs under this schedule are no longer the next
quality intervention. The next accepted work item is the complete
feature-by-feature Prism-versus-UDPipe Development diagnostic, followed only
then by the morphology-specific middle-gradient ablation if the measured
errors justify it.

The all-feature diagnostic is now implemented behind
`--morphology-feature-comparison PATH`. It observes the existing evaluation
predictions rather than running or reimplementing a second Prism pipeline,
validates alignment against the supplied CoNLL-U file, and reports both
systems for every shared feature. The typed JSON contract includes overall and
annotated-token accuracy, count-based per-value precision/recall/F1,
contribution to incorrect complete bundles, and Rare/OOV slices derived from
Prism's training-frequency profile. Console output now labels the columns
explicitly as `Prism`, `UDPipe`, and `Prism-UDPipe`; another external system
can receive its own display name through a CLI option. The selected UD output
policy is stored with the report, so
canonical and explicit treebank-convention comparisons cannot be confused.
Focused accumulator and CLI tests pass.

The first canonical Bokmål report is complete for the selected twelve-epoch
isolated direct-bundle checkpoint. Prism has 1,442 wrong complete morphology
bundles versus UDPipe's 702. Its 2,159 individual feature errors exceed
UDPipe's 1,387 by 772. `Gender` contributes 556 of the excess, `Number` 108,
and `Definite` 66; together these three features explain 94.6% of the excess
individual feature errors. Prism nevertheless leads overall accuracy for
`Case`, `Mood`, `NumType`, `Poss`, and `Tense`, ties `Reflex`, and leads nine
of eighteen features on the OOV slice. This justifies a targeted
morphology-specific intervention rather than a wholesale decoder replacement.

The canonical Nynorsk report is also complete. Prism has 1,931 wrong bundles
versus UDPipe's 1,356 and 2,496 individual feature errors versus 1,909.
`Gender`, `Number`, and `Definite` contribute excesses of 315, 216, and 86;
their combined 617 exceeds the net 587-feature deficit because Prism recovers
30 errors across its stronger features. The `Number` and `Definite` overall
gaps are largely the already documented Nynorsk annotation convention:
canonical Prism emits unsupported `Number=Sing` and `Definite=Def`, while its
annotated `Definite` accuracy is slightly higher than UDPipe's. Those
suppressions remain the responsibility of the explicit treebank output
policy. `Gender` is the remaining shared neural bottleneck, while Prism already
leads Nynorsk overall accuracy for `Abbr`, `Degree`, `Mood`, `Person`, `Tense`,
and `VerbForm`.

The joint all-feature diagnostic therefore authorizes the next
morphology-specific middle-gradient ablation. Its design must widen direct
bundle supervision through morphology-owned parameters without exposing the
Backbone, shared representation, UPOS path, or lemma path to the previously
observed gradient conflict. Acceptance remains the complete joint
Bokmål/Nynorsk canonical all-task and Rare/OOV gate.

That middle-gradient ablation is now implemented. The typed
`MorphologyBundleLossGradientScope` exposes `full`, `morphology`, and
`residual-only`. The new `morphology` path recomputes the morphology adapter,
independent feature heads, and structured decoder from a detached shared token
representation, consumes detached UPOS evidence, and keeps the bundle residual
projection trainable. Value-preserving surrogate tensors keep its direct-loss
candidate scores numerically identical to the normal forward scores even with
dropout. The Backbone, shared projection, character fusion, UPOS head, lemma
path, and reranker refinement gates receive no direct bundle gradient.

The removed legacy isolation switch has no compatibility alias. New runs use
`--morphology-bundle-loss-gradient-scope residual-only`. Checkpoint metadata stores the
resolved string scope, while old checkpoints remain evaluation-compatible
because the setting is training-only and adds no parameters. Parameter-level
tests cover all three scopes, score equality, dropout, and CLI validation.
Implementation alone makes no quality claim; the fixed training
run and canonical per-standard reports are tracked as separate gates.

The fixed joint training run is now complete after approximately 2 hours 11
minutes 35 seconds. Epoch 12 again produced the lowest combined Development
loss. The 70,068,462-byte checkpoint is
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-w010-e12-weighted/best.pt`
and its metadata records gradient scope `morphology`.

The candidate's combined Development loss is 0.112124, between `full` at
0.112011 and `residual-only` at 0.113910. Its bundle loss of 0.126770 is also
between 0.112506 and 0.132241, confirming that the wider morphology path
receives useful direct supervision. Morphology loss is 0.008535 versus
0.008133/0.008676. At the same time, its UPOS loss/accuracy of
0.046252/98.8672% and lemma-rule loss/accuracy of 0.044659/98.8648% are the
best of all three scopes. This is the intended training-level compromise, not
yet a selection result. Separate canonical Bokmål and Nynorsk evaluation
remains mandatory before the candidate can replace `residual-only`.

Canonical Bokmål evaluation is now complete with the fixed checkpoint-derived
logit correction at strength 1.0. The middle scope reaches 98.9991% UPOS,
96.1588% UFeats, and 98.9634% Lemmas. Versus `residual-only`, that is
+0.0137, +0.1237, and -0.0247 percentage points respectively; it recovers 45
complete morphology bundles. OOV morphology micro F1 improves by 0.3305
points, while Rare morphology falls by 0.0945 points and OOV lemma end-to-end
falls by 0.3562 points. The candidate remains 0.5802 UFeats points below
`full` and 1.9110 points below UDPipe. Gender remains the dominant deficit,
although its OOV accuracy improves by 0.9619 points over `residual-only`.
This is promising but not sufficient for selection; the matching Nynorsk
evaluation is still mandatory.

The morphology class weights stored in the three scope checkpoints are
numerically identical. The established correction strength 1.0 therefore
remains fixed for this controlled comparison. It must not be retuned on the
Bokmål result alone.

The canonical Nynorsk report completes the middle-gradient gate. The candidate
reaches 98.7136% UPOS, 93.9104% UFeats, and 98.5856% Lemmas. Relative to
`residual-only`, these improve by 0.0512, 0.0896, and 0.0416 percentage
points, or 16, 28, and 13 correct predictions. Rare/OOV lemma also improves by
0.2925/0.0394 points. Rare morphology falls by 0.2047 points, OOV morphology
falls by 0.0124 points, and Rare/OOV UPOS falls by 0.1254/0.0789 points.

Jointly across Bokmål and Nynorsk, the candidate gains 73 exact UFeats
bundles, 21 UPOS predictions, and a net four Lemma predictions, but shifts
some task-specific slice metrics: Rare morphology regresses on both standards,
all four Rare/OOV UPOS slices regress, and Bokmål OOV lemma loses ten correct
predictions. This initially suggested rejection, but exact slice UFeats had
not yet been available. The selection decision is therefore provisional until
the new target metric is compared on both written standards.

The first exact frequency-slice UFeats report is now recorded for the
`morphology` candidate on Bokmål: Rare UFeats is 88.7619% and OOV UFeats is
87.3174%, representing 354 and 356 wrong complete bundles. Rare plus OOV are
16.38% of Development tokens but contribute 50.82% of all wrong bundles. This
confirms that complete-bundle errors are strongly concentrated away from
frequent forms and that morphology micro F1 alone understated the problem.
The matched `residual-only` Bokmål report reaches 89.3016% Rare UFeats and
85.9637% OOV UFeats. `morphology` consequently loses 17 exact Rare bundles but
gains 38 exact OOV bundles. Thirty-eight of its 45 overall additional correct
bundles are OOV; the preliminary claim that its gain mostly favored frequent
tokens was wrong. Its rejection is withdrawn while the equivalent Nynorsk
exact-slice comparison remains outstanding. Until that gate is complete,
`residual-only` remains the current reference rather than a newly reconfirmed
winner.

The Nynorsk `morphology` slice report now records 86.3769% Rare UFeats and
83.3202% OOV UFeats, corresponding to 326 and 423 wrong complete bundles.
Rare/OOV comprise 15.77% of Nynorsk Development tokens but 39.36% of all
wrong bundles. The equivalent `residual-only` report reaches 86.2934% Rare
UFeats and 83.3596% OOV UFeats, so `morphology` gains two Rare bundles and
loses one OOV bundle on Nynorsk.

The complete joint gate selects `morphology` as the new compact Norwegian
reference. Across Bokmål and Nynorsk it gains 37 exact OOV UFeats bundles
while losing 15 Rare bundles; on the complete splits it gains 73 UFeats, 21
UPOS, and a net four Lemma predictions. The OOV trade is +37 UFeats against
-6 UPOS and -9 Lemmas. This is accepted because complete UFeats is the
demonstrated remaining quality deficit, transfer to supervised-training OOV
forms is the more important LexKeep guardrail, five of six complete-split task
metrics improve, and the scope adds no parameters or inference work. The Rare
regression remains a tracked weakness. `residual-only` is retained as the
protected-gradient control and `full` as the morphology upper control; both
test splits remain untouched.

Against reproduced UDPipe 2.17, the selected reference leads UPOS by 0.0494
points on Bokmål and 0.1408 points on Nynorsk. It trails UFeats by
1.9110/1.7504 points and Lemmas by 0.0110/0.2432 points. Bokmål Lemmas is
therefore effectively tied; complete UFeats remains the primary gap and
Nynorsk Lemmas the secondary one.

Exact OOV feature-error attribution is now part of standalone evaluation and
the analysis JSON. Under the selected correction strength `1.0`, `Gender`
accounts for 251 of 538 Bokmål OOV feature errors (46.65%) and 312 of 583
Nynorsk errors (53.52%). `Number`, `Definite`, `VerbForm`, and `Degree`
complete the top five; together these features explain 92.37%/95.21% of all
Bokmål/Nynorsk OOV feature errors. Counts are per feature rather than per
complete UFeats bundle, so a token can contribute more than once. This
diagnostic is independent of the optional UDPipe feature comparison and both
test splits remain untouched.

The Norwegian training CLI now makes the selected policy safe by default:
when direct bundle loss has a positive weight and no scope is supplied, it
resolves to `morphology`. With bundle loss disabled it remains `full`, which
is behaviorally irrelevant because no auxiliary gradient exists. Explicit
scope values and the hidden legacy isolation alias remain reproducible.

### UFeats optimization guardrail

The remaining UDPipe gap does not replace Prism's architectural direction.
Prism keeps its schema-driven per-feature classifiers as the primary
morphology contract. They expose feature-specific probabilities and errors,
support genuinely multi-valued features, and preserve an open path for bundles
that were not observed in the training inventory. The structured decoder,
Top-32 reranker, and direct bundle objective improve consistency around this
contract; UDPipe is an external reference and not an architecture template.

Official `UFeats` remains mandatory because it measures exact correctness of
the complete feature set for one token. It is intentionally harsher than
per-feature metrics: one wrong feature invalidates the whole token. A higher
UFeats result therefore does not prove that a system is better on every
individual feature. The current direct comparison proves UDPipe leads the
historical measured Prism checkpoint on `Gender`, `Number`, and `Definite`.
No complete all-feature head-to-head has yet established the direction for the
other features or for the newly selected isolated checkpoint.

The accepted morphology path is now:

1. retain the twelve-epoch isolated checkpoint after the rejected 30-epoch
   convergence ablation;
2. execute the implemented all-feature Prism-versus-UDPipe Development report
   on the selected checkpoint, including overall, annotated-token, per-value,
   and Rare/OOV views where applicable;
3. if the direct objective remains underfit, widen its gradient only through
   morphology-specific heads and decoder parameters while protecting the
   Backbone, shared and lemma representations;
4. if the correct bundle remains present but ranked incorrectly, test a small
   nonlinear bundle scorer without replacing the independent feature heads or
   their unseen-combination fallback;
5. change individual feature objectives only where the report demonstrates a
   real deficit, and accept a candidate only on the joint Bokmål/Nynorsk
   canonical all-task gate;
6. retrain the architecture-matched Base Teacher before producing the
   expensive silver-label artifact.

This order is intended to improve real canonical predictions. Treebank output
policies remain separately audited convention translations and cannot count
as evidence that the neural model itself improved. The official test splits
remain untouched.

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

## Silver-data training

The first licensed silver source is now fixed:
Språkbanken's `oai:nb.no:sbr-43` NBdigital Bokmål corpus. The source archive
contains 4,807 public-domain Norwegian books and is distributed as CC0. Prism
does not use its historical Oslo-Bergen tagger labels as training targets.
Only the existing word segmentation and explicit `<<<` sentence boundaries
are retained; Prism's accepted Teacher will produce the later pseudo-labels.

The first reproducible preparation layer is implemented in
`prism.data.nbdigital`, `prism.data.silver`, and
`prism.languages.norwegian.prepare_silver_corpus`. It:

- streams the source `tar.gz` without extracting it;
- accepts only `nob` documents with filename OCR confidence at least `0.95`;
- keeps sentences of at most 128 supplied tokens;
- normalizes case and Unicode before deterministic sentence deduplication;
- removes overlap against the training, development, and test splits of both
  Norwegian written-standard profiles for the selected treebank release;
- writes typed JSONL records plus a versioned manifest containing the source
  URL, archive SHA-256, CC0 provenance, counts, and the complete extraction
  policy;
- rejects empty or internally inconsistent artifacts.

The generated corpus remains under ignored `data/processed/`. This stage does
not yet create pseudo-labels or alter a Student checkpoint. The next
implementation boundary is an offline, provenance-carrying Teacher-label
artifact with per-task confidence. Gold and silver examples will then be mixed
explicitly, with Bokmål/Nynorsk gold sampling preserved so this Bokmål-only
source cannot silently erase Nynorsk quality. Development and test examples
will never enter optimization; both standards remain separate acceptance
gates.

The official archive preparation completed in approximately 30 minutes. With
the fixed OCR and length policy it retains 936 documents, 2,542,722 sentences,
and 50,385,644 tokens in an 880-MB JSONL artifact. The source archive SHA-256
is `9d9c48843d4c9ac845ce775d98118bad667452abe259462770f4a975f23ed505`.
This is far too large for an unmeasured full-corpus Teacher and Student run.
Teacher quality must first pass the official UPOS/UFeats/Lemmas comparison;
the later labeling policy must then select a deterministic, documented subset
rather than silently turning all 50 million tokens into one experiment.

### Nynorsk silver source

The second licensed silver source is fixed: Språkbanken's `oai:nb.no:sbr-60`
corpus of legal documents from Norwegian Nynorsk municipalities, distributed
as CC0. Its single JSON member maps a document URN to
`[page_number, language_code, page_text]` entries, so pages are already
language-classified, but carry no word segmentation, sentence boundaries, or
OCR confidence.

The preparation layer therefore adds two language-independent components with
focused tests. `prism.data.segmentation` implements a versioned
(`prism-rule-segmentation-v1`), deterministic, precision-first sentence
extraction: OCR line-wrap and hyphenation merging, abbreviation- and
ordinal-protected sentence splitting, span-based tokenization with exact
spacing, and conservative quality filters that discard headers, tables,
dot-leader artifacts, quoted fragments, and low-letter-ratio lines. The
Norwegian abbreviation inventory and predeclared filter defaults live in
`prism.languages.norwegian.silver_extraction`. `prism.data.sakspapir` streams
the 897-MB JSON object incrementally, filters pages to `nno`, restores page
order, and deduplicates against both Norwegian gold treebanks. This is an
explicit offline data-preparation policy; runtime raw-text tokenization
remains a separate later product decision.

`prepare_silver_corpus` now selects the source through `--source
{nbdigital-nob,sakspapir-nno}` with backward-compatible Bokmål defaults. The
completed Nynorsk run took approximately 2 minutes 8 seconds and retained
32,406 documents, 2,012,251 sentences, and 37,126,629 tokens in a 725-MB
JSONL artifact under `data/processed/sakspapir-nno/`, validated against its
manifest. The source archive SHA-256 is
`fac1f1f3a7409ad4933e282fda1f416fc1f4d2c80fa8acf98d7de1eb815e343e`. The
manifest records the complete extraction policy including the abbreviation
inventory. Both written standards now have comparable CC0 silver volumes
(50.4M Bokmål and 37.1M Nynorsk tokens), so the later gold/silver mixing
policy does not have to rely on Bokmål-only silver text. No pseudo-labels
have been generated from either source yet.

The source behind that prepared corpus is Språkbanken resource
[`oai:nb.no:sbr-60`](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-60/):
50,000 OCR-derived municipal documents and approximately 127 million words,
of which about 88.5 million are classified as Nynorsk; the corpus is CC0 and
every page carries a language classification. Its JSON source adapter, the
shared UD-overlap and deduplication gates, and the explicit filtering to
Nynorsk-classified pages are implemented and produced the validated artifact
above.

Two other resources remain separate from that running-text contract:
Språkbanken's mixed-language
[`oai:nb.no:sbr-34`](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-34/)
public-domain books are CC0 but require reliable language identification and
stronger OCR/orthography controls; the
[`oai:nb.no:sbr-65`](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-65/)
Nynorsk pronunciation lexicon is also CC0 and supplies inflected forms,
lemmas, and lexical features, but no sentence context. The latter is a
possible later lexical Gender/OOV supervision source, not interchangeable
with Teacher-labeled silver sentences. Norsk ordbank Nynorsk is not classified
as public domain here: its catalogue license is CC BY and therefore needs a
separate provenance and redistribution decision before use.

### Architecture-matched Base Teacher gate

The architecture-matched Base Teacher completed training in approximately
2 hours 12 minutes. Loss selection chose epoch 3; patience 4 stopped the run
after epoch 7. The checkpoint is
`runs/no-teacher-base-character-cnn-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-w010-e12-weighted/best.pt`.
Canonical evaluation with the fixed full logit correction records, per
written standard (Bokmål/Nynorsk): 99.1311%/98.8288% UPOS, 96.5383%/94.8512%
UFeats, and 98.8067%/98.6624% Lemmas.

The predeclared labeler gate is **not passed**. On Nynorsk the new Teacher
clearly beats the selected Student (+0.16 UPOS, +0.77 UFeats, +0.11 Lemmas
points) and the corrected historical control. On Bokmål it loses to its own
compact Student on UFeats (96.5383% versus 96.6372%) and Lemmas (98.8067%
versus 98.9304%), and to the historical control on UPOS and Lemmas. A labeler
that trails the Student on the primary Bokmål quality metric cannot serve as
its Bokmål pseudo-label source.

The training curve identifies the likely cause rather than a capacity limit:
after the loss-selected epoch 3, discrete Development quality kept improving
through epoch 7 (UPOS 98.99% to 99.08%, lemma-rule 98.82% to 99.07%, Gender
annotated 90.53% to 92.97%) while only the loss worsened, which measures
growing overconfidence rather than worse decisions. Loss-based checkpoint
selection therefore discards the strongest discrete teacher epochs, exactly
as previously observed for both historical Base Teachers.

The accepted next step is a predeclared selection-policy ablation instead of
an immediate NorBERT4-large run. The typed
`CheckpointSelectionMetric` now offers `development-loss` (unchanged default)
and `development-task-accuracy`, the mean of UPOS accuracy, lemma-rule
accuracy, and the newly implemented exact complete-morphology-bundle accuracy
over all Development tokens. The bundle-exact metric mirrors the official
UFeats definition and is computed inside the normal evaluation accumulators;
training prints it every epoch. The resolved metric is stored in checkpoint
`training_config`, the CLI exposes it as `--checkpoint-selection-metric`, and
early stopping counts epochs without improvement on the selected metric. The
discrete policy is justified for the teacher/labeler role only: a labeler is
judged by its decisions, temperature calibration later repairs overconfidence
without changing argmax decisions, and the shipped Student keeps loss
selection unless a separate controlled ablation ever argues otherwise. The
Base rerun with discrete selection must pass the same two-standard gate;
NorBERT4-large as labeler remains authorized if it fails again.

The discrete-selection Base rerun is complete and **passes the labeler gate
decisively**. With `--checkpoint-selection-metric development-task-accuracy`,
every epoch produced a new best; epoch 12 was selected after approximately
3 hours 45 minutes at
`runs/no-teacher-base-character-cnn-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-task-acc-e12-weighted/best.pt`.
Development bundle-exact accuracy rose monotonically from 89.93% to 96.73%;
the previously loss-selected epoch 3 corresponds to 94.36%, so loss selection
had discarded 2.37 points of discrete morphology quality in the identical
training trajectory. UPOS and lemma-rule accuracy also peaked at epoch 12,
confirming that the discrete gain is not traded against the other tasks.

Canonical evaluation with the fixed full logit correction records
(Bokmål/Nynorsk): 99.3126%/98.9248% UPOS, 98.1440%/95.5904% UFeats, and
99.2466%/98.9056% Lemmas. Against the selected Student this is +0.34/+0.26
UPOS, +1.51/+1.51 UFeats, and +0.32/+0.35 Lemmas points — every cell of the
two-standard gate passes. Against the loss-selected sibling checkpoint the
discrete selection alone contributes +1.61/+0.74 UFeats points. Against
reproduced UDPipe 2.17, the Teacher now leads every canonical Bokmål metric
(+0.36 UPOS, +0.07 UFeats, +0.27 Lemmas) and leads Nynorsk UPOS and Lemmas
(+0.35/+0.08) with canonical UFeats effectively tied at -0.07 points before
the separately audited Nynorsk treebank output policy. Rare/OOV slices also
clear the Student by wide margins (Bokmål OOV lemma end-to-end 95.16% versus
the Student's 93.69%).

This checkpoint is therefore accepted as the silver labeler source, and a
NorBERT4-large run is no longer required for that role. The remaining
prerequisites before pseudo-labeling stay unchanged: per-task temperature
calibration of this Teacher on Development, then the offline
provenance-carrying label artifact with per-task confidences and the
two-teacher agreement control.

### Teacher temperature calibration

Per-task-head temperature scaling is implemented as a language-independent
mechanism (`prism.training.calibration`) with a Norwegian CLI
(`prism.languages.norwegian.calibrate_baseline`). The fit is a deterministic
two-stage grid search over the log-temperature minimizing the development
negative log-likelihood of exactly the training objective per head:
Cross-Entropy for UPOS, lemma rules, and exclusive morphology features,
Binary Cross-Entropy over the real value logits of multi-valued features.
Calibration consumes the corrected logits, so the artifact is only valid for
its recorded correction strength. Temperatures never change argmax
decisions; a focused test asserts this invariance. The versioned
`TaskTemperatureCalibration` JSON artifact records all twenty temperatures,
checkpoint provenance, language tags, correction strength, and per-head
NLL/ECE reports before and after.

The accepted labeler is calibrated on the combined Bokmål and Nynorsk
development splits with correction strength 1.0; the artifact is
`calibration-corrected.json` beside the checkpoint. Every head was
overconfident, exactly as the discrete selection predicted: temperatures
range from 1.55 (`Case`) to 3.06 (`Foreign`), with UPOS at 2.48, lemma rules
at 1.91, and `Gender` at 2.63. Calibration roughly halves every head's NLL
(UPOS 0.0920 to 0.0460; Gender 0.0844 to 0.0443) and improves Expected
Calibration Error by up to an order of magnitude (UPOS 0.0076 to 0.0011;
Gender 0.0097 to 0.0008). The only nominal regression is `Polarity` ECE at
the 1e-5 noise floor. Confidence-filtered silver labeling can therefore
derive its per-task thresholds from calibrated probabilities; the remaining
prerequisite is the offline label artifact itself.

### Silver label artifact

The offline labeling path is implemented. The language-independent
`prism.training.silver_labeling` module owns the typed contracts
(`SilverSentenceLabels`, sharded `torch.save` storage, and a JSON
`SilverLabelManifest`) and the generation loop; the Norwegian CLI is
`prism.languages.norwegian.label_silver_corpus`. Checkpoint loading for
evaluation-style commands is now shared through
`prism.languages.norwegian.checkpoint_loading`, which the calibration and
labeling CLIs both use.

The artifact deliberately stores **raw calibrated predictions instead of
filtered labels**: per token the complete calibrated UPOS distribution, the
complete calibrated distribution of every morphology feature, and the top-8
lemma-rule distribution in float16, plus the decoded predictions of an
optional second agreement Teacher. Confidence thresholds, two-teacher
agreement, and sentence-discard policies are applied later at training time,
so selection-policy ablations never require relabeling. The manifest records
the source corpus path and SHA-256, the embedded corpus manifest, both
Teacher checkpoint SHA-256 values, the embedded calibration artifact, the
correction strengths, the token budget, counts, and per-shard checksums. A
deterministic pilot subset is selected as the corpus-order prefix until the
token budget is crossed; the crossing sentence is included.

The command validates that the calibration artifact references exactly the
requested checkpoint and correction strength, and that an agreement Teacher
shares the labeler's task schema. Focused tests cover shard and manifest
round trips with checksum verification, count validation, partial-agreement
rejection, budget semantics, and CLI pairing rules.

A 3,000-token smoke run on the Nynorsk corpus with the accepted labeler and
the Bundle-32 Base Teacher as agreement control completed end to end: 146
sentences, 3,019 tokens, verified checksums, probability rows summing to
one, calibrated mean UPOS confidence 0.9876, two-teacher UPOS agreement
99.44%, lemma top-1 agreement 98.84%, and 99.43% of the lemma probability
mass inside the stored top-8. Storage is approximately 757 bytes per token
(0.76 GB per million tokens); per-record tensor overhead dominates short
sentences, so batch-level tensor packing is a noted optimization if the
complete 87M-token corpus is ever labeled. The historical character-CNN
Teacher directory was removed in an earlier cleanup, so the separately
trained Bundle-32 Base Teacher is the designated agreement control.

Both 1M-token pilot labeling runs are complete in approximately 29 minutes
combined: Nynorsk retained 54,463 sentences / 1,000,005 tokens and Bokmål
47,613 sentences / 1,000,002 tokens, three shards each under
`labels-pilot-1m/` beside their corpora. Full-artifact validation passes:
every shard checksum verifies, probability rows sum to one, and the lemma
top-8 captures 99.3% of the probability mass in both artifacts. Calibrated
mean UPOS confidence is 0.9839 (nno) and 0.9738 (nob); the lower Bokmål
values are consistent with the pre-1960 book language sitting further from
the UD training distribution than modern administrative prose. Two-teacher
agreement (labeler versus Bundle-32 control): UPOS 98.96%/98.12%, lemma
top-1 98.51%/98.48%, complete decoded morphology bundle 95.94%/93.92%, and
full-token agreement across all three task groups 94.30% (nno) / 92.40%
(nob). Agreement-based masking alone therefore retains roughly 92–94% of
silver tokens with two independent teachers concurring on every decision.

A cross-corpus duplicate check over both complete prepared corpora found
291 shared sentences / 3,408 tokens (0.009% of Nynorsk tokens) — negligible
double weighting that requires no action; the number is recorded here so
the question stays answered.

### Gold+silver training integration

The training integration is implemented. `prism.training.silver_batches`
loads a silver corpus prefix aligned with its label artifact and applies the
predeclared v1 filter policy at load time: per task group (UPOS, complete
decoded morphology bundle, lemma top-1) a token keeps silver supervision
only when both teachers agree, an optional calibrated-confidence floor can
tighten this (`--silver-minimum-confidence`, default off), and sentences
with more than 30% masked tokens are discarded. `prism.training.
silver_training` implements the soft-target KD objective: cross-entropy
against the complete calibrated UPOS and exclusive-morphology
distributions, soft binary cross-entropy for multi-valued features, and
cross-entropy against the renormalized lemma top-8. `train_mixed_token_task
_epoch` interleaves gold batches (with their unchanged supervised + DKD
objective) and silver batches in a seeded random order, so neither regime
dominates the end of an epoch; the scheduler counts both batch kinds. The
silver configuration and retained counts travel in the checkpoint under
`silver_training`.

The CLI accepts repeated `--silver-corpus`/`--silver-labels` pairs plus
`--silver-loss-weight` (predeclared first candidate 0.5),
`--silver-maximum-masked-ratio`, and the ablation switch
`--silver-disable-agreement-filter`. Loading both 1M pilots takes about
2.5 minutes per source and retains 53,460/54,463 Nynorsk sentences (98.2%,
989,589 tokens; masked upos 0.93%, morphology 3.79%, lemma 1.40%) and
45,466/47,613 Bokmål sentences (95.5%, 968,540 tokens; masked upos 1.28%,
morphology 5.09%, lemma 1.31%) — approximately 1.96M silver tokens beside
the 490k gold tokens. Focused tests cover the agreement masks and sentence
discard on synthetic artifacts, alignment validation, soft-loss behavior
including fully masked batches, deterministic interleaving, and CLI pairing
rules. The predeclared objective ablations remain gold+DKD (fixed
reference), gold+silver-hard, gold+silver-soft, and gold+DKD+silver-soft.

### First silver Student (1M pilot, gold+DKD+silver-soft)

The first silver training run is complete after approximately 5 hours
43 minutes: twelve epochs of gold+DKD (task-accuracy Teacher as the DKD
source) interleaved with both 1M silver pilots at silver loss weight 0.5,
agreement-only filtering, and dual-best selection. The run is
`runs/no-student-silver-pilot-1m-w050-e12-weighted/`. The loss/discrete
divergence previously seen only in Teachers appeared in the Student for the
first time: loss selected epoch 9 while development bundle-exact accuracy
kept rising through epoch 12, so the dual-best mechanism captured both
candidates in one run as designed.

Canonical evaluation with the fixed full logit correction
(Bokmål/Nynorsk):

| Checkpoint | UPOS | UFeats | Lemmas |
| --- | ---: | ---: | ---: |
| E9 loss-selected | 99.0184/98.7552 | 96.3760/94.3872 | 98.9249/98.7488 |
| **E12 task-accuracy** | **99.0679/98.7712** | **96.8820/94.4384** | **99.0294/98.7200** |

The task-accuracy checkpoint passes the complete two-standard gate against
the selected production Student on **every** headline cell: +0.10/+0.10
UPOS, +0.24/+0.36 UFeats, and +0.10/+0.16 Lemmas points. Rare/OOV slices
also improve broadly (Bokmål Rare lemma end-to-end 97.87%, OOV morphology
micro F1 94.96%), and Bokmål annotated `Gender` reaches 94.26% — the
predicted context-to-lexicon transfer from silver text. The loss-selected
sibling wins only marginal slices (Nynorsk Lemmas +0.03, Bokmål Rare lemma
+0.13) but loses Bokmål UFeats by 0.51 points; the discrete selection
policy therefore wins the predeclared Student selection ablation in the
silver regime. Its known cost — slightly worse raw NLL — remains a
calibration-stage concern, and Student temperature calibration is still
pending.

Against reproduced UDPipe 2.17, the E12 candidate now leads Bokmål UPOS
(+0.12), Bokmål Lemmas (+0.05, the first Student lead on that metric),
and Nynorsk UPOS (+0.20); Nynorsk Lemmas trails by 0.11 and canonical
UFeats by 1.19/1.22 points (down from 1.43/1.58). The register question is
answered empirically: administrative Nynorsk silver improved news-register
Nynorsk UFeats by +0.36 points, so the sakspapir source is productive.
The 1M dose therefore validates the recipe end to end; the predeclared 5M
dose step is authorized. Both official test splits remain untouched.

The 5M-token labeling runs for both standards completed in approximately
2 hours 24 minutes combined: Nynorsk 276,273 sentences / 5,000,009 tokens
(14 shards), Bokmål 258,197 sentences / 5,000,007 tokens (13 shards), under
`labels-pilot-5m/` beside their corpora. Full validation passes with
verified checksums. Label quality is stable across the five-fold larger
prefix: calibrated mean UPOS confidence 0.9841/0.9790 and two-teacher UPOS
agreement 98.94%/98.66% (Bokmål agreement actually improves over its 1M
prefix at 98.12%), so no quality drift appears in later corpus regions.
The 5M training run is authorized but has not started.

The task-accuracy checkpoint
`runs/no-student-silver-pilot-1m-w050-e12-weighted/best-development-task-accuracy.pt`
is **selected as the new compact Norwegian Student reference** after passing
the complete two-standard canonical gate on every headline metric. The
previous gold+DKD Student remains the fixed no-silver control for the dose
ladder. Student temperature calibration remains pending and is required
before any shipped confidence claim; the discrete selection's higher raw
NLL is the documented reason. The remaining attribution ablations
(gold+silver without gold DKD, and hard-label silver) stay predeclared but
must not block the dose ladder.

### 5M silver Student and the silver-regime output policy

The 5M training run completed in approximately 14 hours 20 minutes; early
stopping ended it after epoch 8 with the loss-selected checkpoint at epoch
4 and the task-accuracy checkpoint at epoch 7
(`runs/no-student-silver-pilot-5m-w050-e12-weighted/`). Development
bundle-exact accuracy rose to 96.10% (1M run: 95.56%) and plateaued at
epochs 7–8, confirming that the dose was exhausted rather than the
schedule. The task-accuracy checkpoint again beats the loss-selected
sibling across the canonical gate.

Canonical evaluation at the historical correction strength 1.0 initially
showed a UFeats regression versus the 1M Student despite the higher
uncorrected bundle-exact accuracy. The predeclared correction grid
(0/0.25/0.5/0.75/1.0), re-run on Development for the new silver regime,
resolves this: large-scale soft-KD from the calibrated Teacher already
recalibrates the morphology logits during training, so the full
class-weight correction now overshoots. **Strength 0.25 is selected as the
silver-regime output policy**: Bokmål UFeats reaches 97.4594% and Nynorsk
94.7360% (versus 96.7115%/94.3264% at strength 1.0), with Bokmål Rare/OOV
morphology micro F1 at 97.26%/95.35%. Gold-regime checkpoints keep their
separately selected strength 1.0; the policy travels with each checkpoint
as before.

The dose-response curve for canonical Bokmål UFeats at each candidate's
selected output policy: no-silver Student 96.6372% → 1M 96.8820% → 5M
**97.4594%**. Against reproduced UDPipe 2.17 the 5M Student (with its
selected policy) now stands at +0.18 UPOS, **−0.61 UFeats**, and +0.18
Lemmas on Bokmål, and +0.23 UPOS, −0.92 canonical UFeats, and −0.07 Lemmas
on Nynorsk — the Bokmål UFeats gap has more than halved within the pilot
ladder (1.43 → 0.61). Both official test splits remain untouched.

The 1M correction grid completes the true dose reading. The 1M checkpoint's
optima are strength 0.5 on Bokmål (97.1982%) and 0.75 on Nynorsk
(94.6144%). The optimal strength therefore shifts monotonically with the
silver dose — 1.0 (no silver) → 0.5 (1M) → 0.25 (5M) — which cleanly
confirms the mechanism: large-scale soft-KD from the calibrated Teacher
replaces the external class-weight correction from inside. At optimal
policies the curve is logarithmic, not accelerating: Bokmål +0.56 then
+0.26 per 5× step, Nynorsk +0.53 then +0.12. Nynorsk lags Bokmål in the 5M
step, which arms the predeclared register-diversification rule for any
further Nynorsk silver.

The decisive reading is the labeler ceiling: the labeler Teacher stands at
98.1440%/95.5904% UFeats, so the Student has closed 69% of its distance to
its own label source (1.50 points at no-silver, 0.68 at 5M). The flattening
dose curve is therefore approaching the Teacher's own quality, and further
same-label doses buy little. The next binding constraint is labeler
quality again, exactly as the silver policy anticipated: NorBERT4-large as
labeler remains authorized, and a silver-boosted Base Teacher (the
predeclared noisy-student iteration) is the alternative ceiling lift. The
10M same-label dose step is deprioritized in favor of a ceiling lift; it
remains available as a cheap fallback.

### Alternate teacher backbone support

`LanguageProfileSpec` now carries `alternate_teacher_backbones` plus a
`backbone_for_model_id` resolver, and both Norwegian profiles register the
pinned `ltg/norbert4-large` backbone (revision
`49475ca0e59cc5db6ef2c762384b2a916ca8ead0`) as a teacher-only alternative;
`backbone_for_role` keeps returning Base so existing behavior is unchanged.
Checkpoint loading — the shared loader, `evaluate_baseline`, and the
distillation-teacher loader — resolves the backbone from the checkpoint's
stored model ID among the profile's known role backbones instead of
assuming the role default, with the pinned-revision check retained. The
training CLI exposes `--teacher-backbone {base,large}` for teacher-role
runs only. Profile-resolver, pinning, and CLI tests cover the contract.

### Large Teacher (ceiling lift)

The NorBERT4-large Teacher completed all twelve epochs in approximately
7 hours 40 minutes with task-accuracy selection choosing the final epoch
(`runs/no-teacher-large-character-cnn-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-task-acc-e12-weighted/best.pt`,
approximately 1.48 GB). Fine-tuning was stable despite the small gold set;
development bundle-exact accuracy reached 96.99% versus the Base Teacher's
96.73% and was still rising at the schedule boundary. The run also showed
the strongest loss/discrete divergence yet (loss minimum at epoch 3,
discrete peak at epoch 12): loss selection would have discarded 1.6
bundle-exact points, re-confirming the discrete selection policy for the
teacher role.

Canonical evaluation with full logit correction **passes the labeler gate**:
99.2769%/98.9728% UPOS, **98.3502%/95.8272% UFeats**, and
99.3786%/99.0880% Lemmas (Bokmål/Nynorsk), beating the Base labeler by
+0.21/+0.24 UFeats points with Bokmål annotated `Gender` at 97.57% and
Bokmål OOV lemma end-to-end at 96.26%. Against reproduced UDPipe 2.17 the
large Teacher now leads **every canonical cell on both standards**,
including canonical Nynorsk UFeats (+0.17) without any treebank output
policy: +0.33/+0.40 UPOS, +0.28/+0.17 UFeats, +0.40/+0.26 Lemmas. The only
sub-Base cell is Bokmål UPOS (−0.04 points, roughly 13 tokens), which the
gate tolerates given the across-the-board gains. The new labeler ceiling is
therefore 98.35/95.83 UFeats.

The correction-grid subset confirms the gold-regime output policy: UFeats
rises monotonically 0.5 → 0.75 → 1.0 on both standards (98.3145 → 98.3420
→ 98.3502 Bokmål), so strength 1.0 stays selected for the large Teacher —
consistent with the mechanism, since no silver KD recalibrated its logits.
Per-head temperature calibration on the combined Development splits is
complete (`calibration-corrected.json` beside the checkpoint): every head
is overconfident again (UPOS temperature 2.46, lemma 1.97, Gender 2.88),
NLL roughly halves per head, and Gender ECE improves from 0.0096 to 0.0009.

Both 5M pilots were relabeled with the large labeler (correction 1.0,
calibrated) and the Base task-accuracy Teacher as agreement control
(`labels-pilot-5m-large` beside the old label directories; 276,273/258,197
sentences, roughly 1h35m plus 1h48m). The stronger teacher pair agrees
*more*, not less: masked morphology drops from 3.83%/4.46% to 2.78%/2.88%
(Nynorsk/Bokmål) and sentence retention rises to 98.35%/95.62%. Because both
labelers were trained more independently than the previous pair, higher
agreement indicates convergence on the truth rather than on shared errors —
the Student sees roughly 80k additional morphology tokens per standard with
cleaner targets.

The Student retrain on the large labels
(`runs/no-student-silver-5m-large-labels-w050-e12-weighted`, identical recipe
to the Base-label 5M run, early stop after 11 epochs, roughly 20h30m) confirms
the ceiling thesis. In-loop bundle-exact reaches 96.34% at the task-accuracy
best epoch 10 (Base labels: 96.10% at epoch 7), and productive training lasts
longer — cleaner labels delay saturation. The predeclared correction grid on
the canonical gate again selects strength 0.25 in the silver regime
(Bokmål UFeats 97.7233 → **97.7508** → 97.6573 → 97.4456 → 97.1294 for
0/0.25/0.5/0.75/1.0; Nynorsk peaks flat at 0.25/0.5), and the
task-accuracy checkpoint again beats the loss checkpoint (epoch 7,
97.4236 Bokmål UFeats at 0.25) decisively.

Selected canonical cells versus reproduced UDPipe 2.17 (task-accuracy
checkpoint, correction 0.25):

| Cell | Prism Student | UDPipe 2.17 | Delta |
| --- | ---: | ---: | ---: |
| Bokmål UPOS | **99.1476%** | 98.9497% | +0.1979 pp |
| Bokmål UFeats | 97.7508% | **98.0698%** | -0.3190 pp |
| Bokmål Lemmas | **99.2054%** | 98.9744% | +0.2310 pp |
| Nynorsk UPOS | **98.7936%** | 98.5728% | +0.2208 pp |
| Nynorsk UFeats | 94.8448% | **95.6608%** | -0.8160 pp |
| Nynorsk Lemmas | 98.8128% | **98.8288%** | -0.0160 pp |

Four of six cells now beat UDPipe and Nynorsk Lemmas is within five tokens
of a tie. The label-source swap alone bought +0.29 Bokmål and +0.10 Nynorsk
UFeats points at identical dose and recipe, close to the honest +0.35–0.45
projection. The remaining deficits are Bokmål UFeats (0.32 points, labeler
ceiling 98.35 leaves 0.60 points of headroom) and Nynorsk UFeats
(0.82 points), where the small Nynorsk gain strengthens the standing
register-mix hypothesis: the sakspapir municipal-document domain limits
transfer, so a Nynorsk source diversification (Wikipedia candidate) is armed
alongside the 10M dose step and MiniLMv2 attention distillation.

### Nynorsk Wikipedia silver source (register diversification)

The predeclared register-mix decision rule fired: Nynorsk silver gains stay
behind Bokmål at both the dose and the label-quality step, so the 10M stage
adds a second, thematically broad Nynorsk source instead of more sakspapir
text. `prism.data.wikipedia` ingests Wikimedia `pages-articles` XML dumps
(streamed, bz2 or plain) and converts wikitext with a deliberately
conservative, versioned cleaning policy (`prism-wikitext-plain-v1`):
templates, tables, references, images, headings, lists, and redirect or
non-article pages are dropped; wiki links keep their anchor text; any line
that still carries markup residue after cleaning (including bare URLs) is
discarded rather than repaired. Surviving paragraphs run through the same
`prism-rule-segmentation-v1` extraction, gold-fingerprint exclusion, and
casefolded dedup as sakspapir. `prepare_silver_corpus` gains the source
`wikipedia-nno` (default output `data/processed/wikipedia-nno`); the
manifest records CC-BY-SA-4.0 — the same license category as the UD gold
treebanks, documented as provenance only, since models never redistribute
the text. Because dump pages are ordered by page ID, budget-limited
labeling prefers the oldest, predominantly hand-written articles over
late bot-generated stubs. Test coverage: 262/262 passing.

The prepared corpus holds 1,583,469 sentences and 30,504,613 tokens from
171,881 articles (extraction roughly 70 seconds after a 160 MB download).
Labeling 5M Wikipedia tokens with the large labeler and Base agreement
control showed the highest retention of any silver source so far
(99.20% sentences kept, 2.42% masked morphology versus 2.78% sakspapir
and 2.99% NBdigital): both teachers agree most on modern encyclopedic
prose, so the register-broadening source is also the cleanest one.

### 10M mixed-source Student (register mix + dose doubling)

The 10M-per-standard stage trained on three silver sources — Nynorsk
5M sakspapir + 5M Wikipedia, Bokmål 10M NBdigital — with the unchanged
recipe (`runs/no-student-silver-10m-large-labels-w050-e12-weighted`,
roughly 64k silver batches per epoch, 1d10h training after 5h20m
labeling, early stop after epoch 10, loss best epoch 6, task-accuracy
best epoch 10). In-loop bundle-exact reached 96.66% and annotated
`Gender` 93.42% (5M-large run: 96.34%/92.74%) — the largest single-feature
gain lands exactly on the feature the Wikipedia source was chosen for.
The predeclared grid again selects correction 0.25 on the task-accuracy
checkpoint (Bokmål 97.8471 → **97.9021** → 97.7233 for 0/0.25/0.5;
Nynorsk 95.2800 → **95.3408** → 95.3056), and the loss checkpoint loses
again (97.5749/94.9952 at 0.25).

Selected canonical cells versus reproduced UDPipe 2.17 (task-accuracy
checkpoint, correction 0.25):

| Cell | Prism Student | UDPipe 2.17 | Delta |
| --- | ---: | ---: | ---: |
| Bokmål UPOS | **99.1724%** | 98.9497% | +0.2227 pp |
| Bokmål UFeats | 97.9021% | **98.0698%** | -0.1677 pp |
| Bokmål Lemmas | **99.2301%** | 98.9744% | +0.2557 pp |
| Nynorsk UPOS | **98.8384%** | 98.5728% | +0.2656 pp |
| Nynorsk UFeats | 95.3408% | **95.6608%** | -0.3200 pp |
| Nynorsk Lemmas | **98.8672%** | 98.8288% | +0.0384 pp |

Nynorsk Lemmas flipped to a win, so four of six cells now beat UDPipe
and only UFeats remains open on both standards. The register-mix rule
delivered as predicted: Nynorsk UFeats gained +0.4960 points in one
stage (dose alone had produced +0.10) and its UDPipe gap shrank from
0.8160 to 0.3200 points; Bokmål UFeats gained +0.1513 from the dose
doubling, matching the logarithmic dose curve, leaving a 0.1677-point
gap. The remaining planned quality lever is MiniLMv2 attention-relation
distillation (improvement idea 2); further dose (20M) is expected to
contribute only ~0.05-0.1 more per the log curve and is secondary.

### Token-relation distillation (MiniLMv2-style, final planned lever)

`prism.training.relation_distillation` implements the last predeclared
quality measure before the project freezes the benchmark stand. Instead
of hooking into the pinned custom backbone attention code, Prism adapts
MiniLMv2 to the word-aligned pooled backbone states every `TokenTagger`
exposes before its task heads (`encode_pooled_token_states`): that
boundary is architecture- and tokenizer-agnostic, and the configurable
relation-head split (default 8, dividing xsmall 192, base 640, and large
1024 alike) restores MiniLMv2's multi-head relations on top of it. The
loss is the KL divergence between teacher and student relation
distributions (softmax-normalized pairwise similarities), computed with
a finite mask value so padding cannot produce NaNs, averaged over valid
tokens and heads, and backpropagated into the student only.

The objective runs on gold batches inside the existing distilled step:
a second frozen teacher (`--relation-teacher-checkpoint`, intended to be
the large task-accuracy teacher) contributes one extra forward pass per
gold batch (~3% of a mixed epoch), so the cost is small next to silver.
`--relation-distillation-weight` (default 1.0) and
`--relation-head-count` (default 8) complete the CLI; the flags require
a student role, a gold DKD teacher, and silver mixed training, and the
configuration is recorded in the checkpoint under
`relation_distillation`. The mean relation loss is reported per epoch.
Test coverage: 272/272 passing. The final run repeats the 10M
mixed-source recipe plus relation distillation from the large teacher;
its gate decides the shipped benchmark stand.

The relation run
(`runs/no-student-silver-10m-relation-w050-e12-weighted`, weight 1.0,
8 relation heads, large relation teacher) trained stably — the relation
loss fell monotonically from 0.385 to 0.032 — and was killed by the
macOS out-of-memory handler during epoch 12 after 1d18h41m. The
truncation is a documented deviation from the predeclared 12-epoch
schedule but immaterial to selection: both metrics peaked at epoch 10
and had already declined at epoch 11, and both best checkpoints were
safely written at epoch 10. Notably, loss selection and task-accuracy
selection **coincide for the first time** in any silver run — the
relation objective closed the overconfidence divergence — and the
canonical correction optimum shifted accordingly (Bokmål optimum now at
strength 0.0 with 97.8663, Nynorsk at 0.25 with 95.2032).

The gate itself is a null result: at each standard's optimal strength
the relation student trails the 10M mixed-source student by 0.0358
(Bokmål) and 0.1376 (Nynorsk) UFeats points, with UPOS and Lemmas flat
(±0.01). Better internal calibration did not translate into better
canonical quality on this recipe, so the predeclared acceptance rule
rejects the final lever.

## Frozen benchmark stand

The shipped reference is therefore the 10M mixed-source student
**without** relation distillation:
`runs/no-student-silver-10m-large-labels-w050-e12-weighted/best-development-task-accuracy.pt`
(epoch 10, ~70 MB) with morphology-logit-correction strength 0.25.
Against reproduced UDPipe 2.17 (~700 MB) on the Development splits:

| Cell | Prism Student | UDPipe 2.17 | Delta |
| --- | ---: | ---: | ---: |
| Bokmål UPOS | **99.1724%** | 98.9497% | +0.2227 pp |
| Bokmål UFeats | 97.9021% | **98.0698%** | -0.1677 pp |
| Bokmål Lemmas | **99.2301%** | 98.9744% | +0.2557 pp |
| Nynorsk UPOS | **98.8384%** | 98.5728% | +0.2656 pp |
| Nynorsk UFeats | 95.3408% | **95.6608%** | -0.3200 pp |
| Nynorsk Lemmas | **98.8672%** | 98.8288% | +0.0384 pp |

Four of six canonical cells beat UDPipe at one tenth of its size; the
two open UFeats gaps are documented, honest residuals.

### One-shot test benchmark

With the configuration frozen (checkpoint, correction 0.25, all policies
fixed on Development), each system was evaluated **once** on the
untouched test splits (`--split test`; UDPipe via the same
gold-tokenized LINDAT reproduction on byte-identical gold files,
predictions under `runs/udpipe-2.17-251125/ud-current/`):

| Test cell | Prism Student | UDPipe 2.17 | Delta |
| --- | ---: | ---: | ---: |
| Bokmål UPOS | **98.7619%** | 98.5717% | +0.1902 pp |
| Bokmål UFeats | 97.1968% | **97.5906%** | -0.3938 pp |
| Bokmål Lemmas | **98.9755%** | 98.8654% | +0.1101 pp |
| Nynorsk UPOS | **98.7688%** | 98.5993% | +0.1695 pp |
| Nynorsk UFeats | 96.9362% | **97.3842%** | -0.4481 pp |
| Nynorsk Lemmas | **98.6760%** | 98.5630% | +0.1130 pp |

The Development picture replicates exactly: the same four cells win
(UPOS and Lemmas on both standards), UFeats stays the open gap on both.
The UFeats deltas are slightly wider than on Development (-0.39/-0.45
versus -0.17/-0.32), consistent with policies having been selected on
Development plus ordinary split variance; no test-driven adjustment was
or will be made.

### Speed benchmark

`prism.languages.norwegian.benchmark_speed` times the complete decision
(subword tokenization, character batching, device transfer, forward,
logit correction, decoding) on the Bokmål test split (1,939 sentences,
29,966 tokens), warmup 8 batches:

| Configuration | Latency p50 | Sentences/s | Tokens/s |
| --- | ---: | ---: | ---: |
| CPU, batch 1 | 20.5 ms | 45.1 | 696 |
| CPU, batch 32 | 184.4 ms/batch | 165.9 | 2,564 |
| MPS, batch 1 | 31.0 ms | 26.5 | 410 |
| MPS, batch 32 | 83.7 ms/batch | 188.9 | 2,919 |

For interactive single-sentence use the CPU path is the fastest option
(~21 ms per decision; MPS dispatch overhead dominates batch 1), and CPU
throughput of ~2.6k tokens/s tags a book chapter in about two seconds —
the offline deployment target needs no GPU. UDPipe 2 has no comparable
local path on this machine; its LINDAT service round-trip (network and
queueing included, an upper bound only) measured 14.88 s for the Bokmål
test file (~2,014 tokens/s) and 14.10 s for Nynorsk (~1,757 tokens/s).
The deployment contrast stands regardless: a ~70 MB local model with
~21 ms interactive latency versus a ~700 MB model behind a server.

### Runtime tagging API with calibrated confidences

The frozen student is now consumable as a library. Per-head temperature
calibration was fitted on the Development splits at the production
correction strength 0.25 (argmax-invariant, so methodologically clean
after the one-shot test benchmark; artifact
`calibration-corrected.json` beside the checkpoint). The headline
finding: the student is **almost self-calibrated** — temperatures sit
at 1.01–1.14 versus 2.46–2.88 for the gold-only teachers, confirming
that silver soft-target KD recalibrates internally. After scaling, UPOS
ECE is 0.0017 and `Gender` ECE 0.0016; the shipped confidences are
trustworthy.

`prism.data.segmentation.segment_pretokenized_sentences`
(`prism-runtime-segmentation-v1`) adds the recall-oriented runtime
counterpart to the silver extraction: same line merging, protected
boundaries (abbreviations, ordinals), and UD token conventions, but
user text is never dropped — headings and fragments become sentences
and over-long sentences are chunked into policy-sized windows instead
of being discarded.

`prism.languages.norwegian.tagger.NorwegianTagger` ties both together
for applications such as LexKeep: `tag_text` (raw text, runtime
segmentation first) and `tag_pretokenized` (application-supplied
tokens) both return tokens with UPOS, morphology features, lemma, and
calibrated confidences per decision, decoded with the frozen production
policy. Lemma decoding reuses the same `LemmaEditRule.apply` convention
as the UD evaluation and the export artifact. A CPU smoke test on raw
text (3 sentences, 32 ms) reproduced line-wrap merging, abbreviation
and ordinal protection, correct lemmas (bøker→bok, sov→sove), and the
intended confidence behaviour: the all-caps OOV heading token
"KAPITTEL" received a wrong lemma at 0.27 confidence while every
regular decision sat above 0.97 — low confidence flags exactly the
decisions an application should not trust. Test coverage:
298/298 passing (one ExecuTorch lowering test requires the missing
`flatc` binary and is environment-blocked, not code-blocked).

### Book-chapter fixture (idea 19.4, LexKeep-realistic input)

`data/examples/hp7kap1.txt` holds a complete novel chapter exactly as
LexKeep reads it. The first pass exposed the dominant real-world defect:
e-book extraction loses spaces after sentence punctuation
("veien.Et sekund"), which fused 105 tokens (~3%), hid half the sentence
boundaries (113 detected), and produced garbage lemmas — all of it
correctly flagged by low confidences. The runtime segmentation now
restores those spaces deterministically
(`_restore_missing_sentence_spaces`: lowercase letter + terminal
punctuation immediately followed by an uppercase or opening character
never form one token in Norwegian prose; abbreviation-protected
boundaries are still consulted afterwards, so "f.eks.Dette" repairs
without creating a false sentence break). Runtime-only — the frozen
silver extraction policy is untouched.

After the fix the chapter yields 247 sentences and 3,783 tokens in
1.7 s on CPU (~2,250 tokens/s) with zero fused tokens; the low-
confidence tail halved (UPOS below 0.8: 5.4% → 3.0%, lemma 5.2% → 2.9%,
below 0.5: 0.6%). The remaining tail is the honest residual: all-caps
heading words (lemma casing garbage at 0.09–0.30 confidence) and
genuinely rare verb forms (skalv, rakte) — the documented Rare/OOV
weakness, reliably marked by confidence. A LexKeep threshold around 0.8
separates auto-trustable decisions from the ~3% that deserve a fallback.
Test coverage: 299/299 passing.

### Export precision decision and calibrated-probability graph

The export pipeline now bakes the complete decoding policy into the
lowered graph: `--calibration` adds a `CalibratedProbabilityExportLayer`
(per-head temperature scaling, softmax for exclusive heads, sigmoid for
multi-valued features) behind the already-embedded 0.25 logit
correction, so the artifact emits final calibrated probabilities
(`*_probabilities` output names), copies `calibration.json` into the
artifact, and native runtimes need nothing beyond argmax and the 0.5
threshold. The correction and calibration tail always computes in
float32 — both for portable-kernel coverage and numerical safety — and
the parity machinery stays exact through a log/logit shim that maps
probabilities back to decode-equivalent logits.

`--precision` was decided by the predeclared ±0.02 pp gate on the
canonical Development cells:

- **fp16 (everything halved): rejected.** Nynorsk UFeats −0.1024 pp
  (Bokmål −0.0248); the loss concentrates in the fp16 bundle reranker,
  whose `log_softmax` also lacks a portable fp16 kernel at runtime.
- **fp16-backbone (backbone fp16, heads/reranker/correction/calibration
  fp32): accepted.** Worst cell delta −0.0096 pp, Nynorsk UPOS and
  Lemmas bit-identical, two cells marginally above fp32.

The shipped artifact `models/prism-no-0.2.0-fp16` lowers to **43.8 MiB**
(fp32: 84 MiB; the raw fp32 weights are 67.2 MiB — the overhead is
XNNPACK delegate weight duplication). Runtime parity against eager:
decoded predictions identical on both fixture batches, max probability
difference 2.7e-3 (Bokmål) and 1.3e-2 (Nynorsk) — within the
fp16-appropriate tolerance 0.02 (the 5e-3 default remains for fp32
exports). `fixtures.json` (6 MB) is a development artifact for native
parity testing and is not shipped, so the deployable set is roughly
48 MB: program, vocabulary, labels, calibration, manifest. The missing
`flatc` binary was installed (Homebrew flatbuffers), unblocking the
ExecuTorch lowering test: 300/300 passing.

A direct program benchmark (fixture batch of 8 sentences at the fixed
160×96 shapes, 20 timed runs after warmup) then reversed the shipping
decision: the fp32 program needs 257 ms per batch (32 ms/sentence),
the fp16-backbone program **3,668 ms** (459 ms/sentence) — roughly
14× slower. The XNNPACK partitioner does not delegate the fp16
operators, so the halved weights execute in naive portable kernels;
size dropped but speed collapsed. Since offline latency is a core
product goal, **fp32 remains the shipped precision**; fp16-backbone
passed the quality gate but fails the deployment gate. The shipping
artifact is **`models/prism-no-0.2.0`** (fp32, calibrated-probability
graph, 83.9 MiB program): runtime parity max |Δp| 1.63e-3 (Bokmål) and
1.94e-3 (Nynorsk) within the 5e-3 fp32 tolerance, decoded predictions
identical on both fixture batches. The documented next size lever is XNNPACK
**int8 quantization**, which is a first-class delegated path and would
need its own predeclared quality gate (~22 MiB expected). The lever
history that produced this stand: dose ladder 1M→5M→10M (logarithmic),
labeler ceiling lift base→large (+0.29 Bokmål UFeats at fixed dose),
register diversification via Nynorsk Wikipedia (+0.50 Nynorsk UFeats in
one stage), dual-best checkpoint selection, and per-head calibrated,
agreement-filtered soft labels. Attempted and rejected with evidence:
TAKD cascade, more sakspapir dose for Nynorsk, and MiniLMv2-style
relation distillation. The official test splits remain untouched.

That future Student ablation is prepared through an optional dual-best
mechanism: `--secondary-checkpoint-selection-metric` tracks a second metric
in the same deterministic run and writes its best epoch next to the primary
checkpoint as `best-<metric>.pt`, with `checkpoint_selected_by` recorded in
both checkpoints. The secondary metric never influences early stopping or
the primary checkpoint, so historical behavior is unchanged when the option
is omitted. The predeclared Student comparison — loss-selected versus
task-accuracy-selected from one silver-training run, judged on the full
canonical gate including the later calibration metrics — can therefore run
without additional training cost. Both selection candidates share the run's
early-stopping boundary, which follows the primary metric.

## Frozen morphology-head probe

A bounded head-capacity diagnostic is now implemented before the next full
Student experiment. `TokenTaskHeads` and `TokenTagger` expose the exact
`morphology-pre-head` representation after the selected shared Wide-MLP,
character fusion, and morphology adapter. Classification still uses that same
typed boundary, so the probe does not maintain a second approximation of the
production forward path.

`prism.training.morphology_probe` freezes the complete source checkpoint and
trains only three small, schema-driven controls on cached training
representations:

- `linear`: matched independent linear feature heads;
- `shared-mlp`: one additional shared residual `H -> 2H -> H` projection;
- `feature-mlp`: one residual bottleneck MLP per morphology feature.

All controls use the checkpoint's morphology class weights and the requested
logit-correction strength. They report every feature on overall, annotated,
Rare, and OOV Development tokens for each supported written standard. A
versioned optional representation cache is validated against the checkpoint
SHA-256, serialized schema, treebank release, evaluation language tags, and
representation-boundary name. The probe never reads test labels, never
updates the source model, and never writes a replacement checkpoint.

This diagnostic deliberately precedes another multi-hour end-to-end run. A
feature-specific MLP is worth integrating into the production architecture
only if it yields a material, repeatable `Gender` gain on both Bokmål and
Nynorsk without merely moving errors into the other dominant Rare/OOV
features. If neither nonlinear probe clearly beats the matched linear probe,
the next intervention must target representation, supervision, or annotation
contracts rather than adding head complexity.

The first seed-42 probe is complete in 3 minutes 14 seconds, including the
one-time extraction of 489,216 training, 36,369 Bokmål Development, and 31,250
Nynorsk Development token representations. The validated FP32 cache is
439 MB. All three training losses were still decreasing at epoch 8, so this
run is a strong screening signal rather than the final capacity choice.

Both nonlinear probes beat the matched linear control decisively on `Gender`.
The feature-specific MLP gains 3.0878/2.2592 overall percentage points and
9.6642/6.6633 annotated-token points on Bokmål/Nynorsk. Rare gains are
7.0476/6.1011 points and OOV gains 4.6313/3.6672 points. It also improves
`Definite`, `Number`, `PronType`, `VerbForm`, `Degree`, and nearly every other
overall feature. Its only overall regression against linear is Nynorsk
`NumType` by 0.0096 points, equal to three tokens.

The much smaller shared MLP already captures most of this signal. Relative to
the feature-specific MLP it is behind by 0.4949/0.4640 overall `Gender` points,
1.6344/1.5236 annotated points, and 0.9524/1.2537 Rare points. It ties the
feature-specific MLP on Nynorsk OOV and is eight Bokmål OOV tokens better.
Parameter counts are 10,036 for linear, 158,068 for shared MLP, and 678,772
for feature MLP; the feature-specific option would add roughly 2.55 MiB of raw
FP32 parameters over linear heads.

These standalone probes stop before the selected structured decoder,
Bundle-32 reranker, and agreement path. The existing complete production
pipeline therefore remains slightly better than the best standalone probe on
`Gender`: by 0.2062/0.2592 overall points and 1.5319/1.1830 OOV points on
Bokmål/Nynorsk. The result proves that the frozen representation is
nonlinearly informative and that linear separability is a real bottleneck; it
does not yet prove that adding the MLP before the existing structured path
will produce additive gains. A longer cached probe and a second seed are the
cheap confirmation gates before authorizing another full Student training.

The matched seed-42 extension to 16 epochs is complete in another 3 minutes
14 seconds using the cache. Training loss continues to fall for all three
heads, but Development gains have mostly saturated. Versus epoch 8, the
feature-specific MLP gains only 0.1870/0.0416 overall `Gender` points on
Bokmål/Nynorsk. Its Bokmål OOV Gender falls by three correct tokens while
Nynorsk OOV gains eight, so lower training loss is no longer a reliable proxy
for the desired transfer slices.

At epoch 16, feature MLP still beats shared MLP on overall Gender by
0.1705/0.0480 points, annotated Gender by 0.6040/0.0508 points, and Rare
Gender by 0.5397/0.6686 points. Shared MLP is 0.5344 points, or 15 tokens,
better on Bokmål OOV Gender; feature MLP is 0.5521 points, or 14 tokens,
better on Nynorsk OOV Gender. Across all 18 features, feature MLP removes 136
more Bokmål overall errors than shared MLP but adds 12 OOV errors; on Nynorsk
it removes only nine overall and 15 OOV errors. The extra 520,704 parameters
therefore do not yet provide a robust enough advantage to choose
feature-specific heads.

The 16-epoch feature probe nearly reaches the complete selected pipeline on
Bokmål overall Gender, trailing by only seven tokens, but still trails its OOV
Gender by 46 tokens. Nynorsk remains 68 overall and 22 OOV Gender tokens
behind. More same-seed epochs are not the next gate. One matched 16-epoch
second seed will measure whether shared-versus-feature differences are stable;
only then should a full production candidate be selected.

The matched seed-43 run is complete in 3 minutes 15 seconds. It repeats the
large nonlinear gain over the linear control, but it does not establish the
feature-specific MLP as the safer architecture. Relative to shared MLP,
feature MLP improves overall Gender by 0.0880/0.2112 points, annotated Gender
by 0.3375/0.7314 points, and Rare Gender by 0.0635/1.2537 points on
Bokmål/Nynorsk. It simultaneously loses 0.3207/0.4338 OOV Gender points,
equal to nine and eleven additional errors.

Across all 18 features in seed 43, feature MLP removes 56/78 overall and 6/31
Rare errors relative to shared MLP on Bokmål/Nynorsk, but adds 11/18 OOV
errors. Combining both 16-epoch seeds, feature MLP removes 279 overall and 46
Rare feature errors while adding 26 OOV errors. Its additional 520,704
parameters therefore improve mostly in-distribution decisions but fail the
explicit OOV generalization gate.

The next full Student candidate is consequently the smaller post-fusion
shared morphology MLP, not the feature-specific MLP. It adds one generic
residual `H -> 2H -> H` transformation at the `morphology-pre-head` boundary
before the existing independent feature heads, structured decoder, Bundle-32
reranker, and agreement path. This is a candidate selection, not a new
production standard: only a controlled end-to-end training and separate
Bokmål/Nynorsk evaluation may replace the current checkpoint. The
feature-specific probe remains a diagnostic upper-capacity control rather
than a planned production component.

The selected ablation is now implemented as the orthogonal
`MorphologyPreHeadArchitecture` contract. `identity` preserves the exact
historical path and is the fallback for every older format-3 checkpoint;
`shared-mlp` inserts the probe-matched residual `H -> 2H -> H` projection
after character fusion and before every independent morphology head. The
existing structured decoder, Bundle-32 reranker, UPOS branch, and lemma branch
are otherwise unchanged.

The new block is deliberately separate from `TokenTaskHeadArchitecture`, so
future languages and existing task-head variants can reuse the same ablation
without multiplying combined enum values. At Norwegian xsmall hidden size 192
it adds 148,032 parameters, approximately 0.57 MiB in FP32. It is initialized
after all existing modules, preserving bit-identical initialization of every
common parameter under the same random seed. Checkpoints store
`morphology_pre_head_architecture`; training prints it, and evaluation,
and distillation-teacher loading reconstruct it strictly. Strict
`torch.export` parity covers the character-aware,
structured, Bundle-32 path with the new MLP.

The direct bundle-loss `morphology` scope includes the new projection while
continuing to protect the shared representation, Backbone, UPOS, and lemma
from that auxiliary term. Normal supervised morphology loss and morphology
distillation still reach shared upstream parameters, so the end-to-end
acceptance gate must explicitly check that UPOS and Lemmas do not regress.
Implementation alone makes no quality claim; the current selected production
checkpoint remains unchanged.

The controlled twelve-epoch run is now complete. Epoch 12 again has the lowest
combined Development loss, and the resulting 70,661,786-byte checkpoint is
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-w010-e12-weighted/best.pt`.
Relative to the selected `identity` control, bundle loss improves from
0.126770 to 0.118622, a 6.43% relative reduction. Combined loss nevertheless
rises from 0.112124 to 0.114135, UPOS accuracy falls from 98.8672% to
98.8302%, and lemma-rule accuracy falls from 98.8648% to 98.8367%. The
training signal therefore confirms useful morphology-specific capacity but
also the anticipated all-task regression risk. The candidate is not selected
or rejected until separate canonical Bokmål and Nynorsk evaluation covers
complete UFeats, every feature, and Rare/OOV slices under the selected full
logit-correction policy. The current production checkpoint remains unchanged.

The first end-to-end gate is complete on canonical Bokmål with the selected
full logit correction. Relative to the selected `identity` control, the shared
morphology MLP raises UFeats from 96.1588% to 96.6372%, correcting 174
additional complete bundles. Rare morphology micro F1 rises by 0.8338 points.
Gender gains 0.4124 points overall, 1.5012 on annotated tokens, and 1.8095 on
Rare tokens. These are substantial learned-output improvements.

The trade-off is not uniformly positive. OOV morphology micro F1 falls by
0.1582 points and OOV Gender by 0.3919 points. UPOS loses 0.0302 points or 11
correct tokens; Lemmas loses 0.0330 points or 12 correct tokens. Rare/OOV
lemma end-to-end accuracy falls by 0.0318/0.4275 points. Against UDPipe 2.17,
the candidate leads UPOS by 0.0192 points but still trails UFeats by 1.4326
points and Lemmas by 0.0440 points. It therefore closes a material part of the
Bokmål UFeats gap without yet satisfying the OOV and all-task
no-regression ideal. The matched canonical Nynorsk gate remains mandatory
before selection or rejection; the current production checkpoint remains
unchanged.

The matched canonical Nynorsk gate is complete. Relative to `identity`, the
shared morphology MLP raises UFeats from 93.9104% to 94.0768%, gaining 52
complete bundles. Rare/OOV morphology micro F1 gains 0.6893/0.3581 points.
Gender gains 0.1728 points overall, 0.5587 annotated, 0.3343 Rare, and 0.1972
OOV. UPOS falls by 0.0448 points or 14 predictions, Lemmas by 0.0288 points or
nine predictions, and Rare/OOV lemma end-to-end by 0.1672/0.0789 points.

The complete two-standard gate selects the shared morphology MLP as the next
compact architecture reference. It gains 226 exact UFeats bundles, removes
279 individual feature errors, and removes 204 Gender errors across Bokmål
and Nynorsk. It loses 25 UPOS and 21 Lemma predictions. On the combined 5,343
OOV tokens it loses four exact UFeats bundles, adds ten individual feature
errors, and adds six Gender errors, so the selection is not a claim of
uniform improvement. The much larger complete-split and Rare gains are
accepted because UFeats/Gender remains the demonstrated primary quality gap,
while UPOS still leads UDPipe on both written standards. OOV morphology and
lemma generalization are explicit next-intervention guardrails.

The selected checkpoint is now
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-w010-e12-weighted/best.pt`.
Its metadata already reconstructs `shared-mlp` exactly. The generic
model-building defaults and checkpoint-metadata fallback remain `identity`
for compatibility. The Norwegian training CLI now defaults new runs to
`shared-mlp`; passing `--morphology-pre-head-architecture identity`
reproduces the previous control explicitly. Evaluation, Teacher loading,
export, and probes continue to reconstruct the checkpointed value rather than
using the new-training default. Official test splits remain untouched.

### Post-shared-MLP investigation and current roadmap

The read-only task-interaction audit and the first nonlinear scorer ablation
are complete. The selected checkpoint remains unchanged because the scorer
candidate fails the canonical Bokmål gate. The next work now targets the
measured fusion and coverage boundaries rather than adding more Backbone
capacity.

1. Retain the completed task-interaction audit as the diagnostic baseline.
2. Record and reject the completed `compositional-mlp` scorer as a full output
   path. It improves internal candidate Top-1 but regresses final UFeats,
   Rare/OOV morphology, and Lemmas after the current static fusion.
3. Retain the completed **frozen adaptive probability-fusion probe** as a
   rejected diagnostic. All existing model parameters remained frozen and
   only a compact token- and feature-dependent gate learned when to trust the
   independent feature path or the bundle path. It failed the joint
   Bokmål/Nynorsk and Rare/OOV gate.
4. Retain the completed candidate decomposition: Top-64 removes every
   training-seen pruning miss on both standards; never-seen bundles affect
   only one Bokmål and 150 Nynorsk Development tokens.
5. Retain the completed controlled Top-64 reranker ablation as rejected.
   Although it removes all measured Top-32 pruning misses, it regresses
   Bokmål UFeats and Rare/OOV quality and yields only a small Nynorsk gain.
   Defer bounded compositional candidate generation and preserve the selected
   Top-32 inventory plus per-feature confidences.
6. Defer the proposed exact-bundle consistency objective on the **final
   post-fusion probabilities**. It remains a documented architecture option,
   but the completed Top-64 result does not justify another expensive
   architecture run before testing whether broader lexical coverage addresses
   the dominant Gender/Rare/OOV errors.
7. Implement the offline, provenance-carrying Teacher-label artifact and run
   one deterministic, confidence-filtered pilot over the already prepared
   Bokmål source. This is the immediate implementation boundary; it does not
   authorize labeling all 50 million tokens.
8. Add the language-independent JSON source-adapter boundary needed by the
   CC0 Nynorsk municipal corpus, prepare only its Nynorsk-classified content,
   and define a balanced Bokmål/Nynorsk silver sampling policy.
9. Compare the resulting Gold/Silver Student with the selected Gold-only
   Student under the unchanged two-standard, per-feature, Rare/OOV, UPOS, and
   Lemma gate. Silver data are accepted only if Gender improves materially
   without moving losses into other tasks.
10. After morphology stabilizes, evaluate a frozen lemma near-miss reranker.
   Soft UPOS/morphology context is accepted only if it resolves audited errors
   beyond character and edit-rule evidence.
11. Reconsider the final-output bundle objective and NorBERT4-large only if
   the controlled silver-data result exposes a remaining representation or
   structured-decoding limit.

The audit is exposed by `evaluate_baseline
--task-interaction-audit --task-interaction-gradient-batches N` and is
implemented generically in
`python/src/prism/training/task_interaction_audit.py`. It performs no optimizer
step and reports checkpoint-weighted supervised gradients only. The selected
shared-MLP checkpoint produced:

| Development split | Final bundle errors | Missing candidate | Ranking | Refinement |
| --- | ---: | ---: | ---: | ---: |
| Bokmål | 1,223 | 72 | 930 | 221 |
| Nynorsk | 1,851 | 782 | 880 | 189 |

For covered Bokmål errors, 74.11% place the gold bundle in the first two
candidates and 92.88% in the first five. Nynorsk reaches 71.19% and 89.06%.
This confirms that the existing candidate evidence commonly contains the
correct answer but ranks it too low, especially on Bokmål. Nynorsk additionally
retains a material candidate-coverage problem that a scorer alone cannot
solve.

The gradient result does not justify immediately combining a conflict method
with the scorer change. Bokmål mean cosines are positive in every measured
group and pair, with conflict rates between 0% and 18.75%. Nynorsk is also
positive on average; its shared projection has a 56.25% UPOS-versus-Lemma
conflict rate on 16 sampled batches, but Morphology-versus-Lemma is 25% there
and 18.75% in the Backbone. This isolated signal remains a later,
separately-trained diagnostic rather than a selected optimization policy.

Lemma has meaningful reranking headroom but no evidence yet that direct
UPOS/morphology coupling is the cause. Gold-rule Top-1/Top-2 is 99.01%/99.78%
overall and 93.37%/98.24% on Bokmål OOV; Nynorsk is 98.64%/99.63% overall and
92.53%/97.38% on OOV. A future lemma experiment must first attribute these
near-miss errors to contextual evidence versus character/edit-rule evidence.

The audit-motivated scorer ablation is implemented as
`--morphology-bundle-scorer-architecture compositional-mlp`. It composes each
candidate representation from learned schema-derived UPOS and feature-label
embeddings and compares it with a nonlinear token query. The final query
projection is zero-initialized, so the additional residual begins at exactly
zero over the existing evidence. `linear` remains the default and
backward-compatible checkpoint fallback. For the current 185-candidate,
`H=192` Norwegian contract, the reranker grows from 35,723 to 89,298
parameters: an increase of 53,575 parameters, or about 209 KiB of raw FP32
weights. Both variants pass strict `torch.export`.

The matched Bokmål result rejects the scorer as the new default:

| Metric | Selected linear scorer | Compositional scorer | Change |
| --- | ---: | ---: | ---: |
| UPOS F1 | 98.9689% | **99.0019%** | +0.0330 pp |
| UFeats F1 | **96.6372%** | 96.5960% | -0.0412 pp |
| Lemmas F1 | **98.9304%** | 98.8644% | -0.0660 pp |
| Rare UFeats F1 | **90.9841%** | 90.6032% | -0.3809 pp |
| OOV UFeats F1 | **86.7474%** | 86.2487% | -0.4987 pp |

The result is not a capacity failure. Candidate Top-1 on covered tokens rises
from 97.3388% to 97.4773%, and ranking errors fall from 930 to 871. At the
same time refinement errors rise from 221 to 277 and final errors rise from
1,223 to 1,238. The current token-invariant residual gates therefore suppress
or reverse useful rank evidence. A hard candidate Top-1 diagnostic reaches
96.7885% over all Bokmål tokens, exposing 0.1513 points over the selected
final output, but is not a production solution because it removes the
independent path's open-combination fallback.

The completed probe consequently changed only fusion. It mixed probabilities
as `p_final = (1 - g) * p_feature + g * p_bundle`, with `g` predicted from
inference-available model confidence, margins, agreement, and the morphology
token representation. It was trained on the training split with the selected
checkpoint frozen. A planning expectation of roughly 0.15--0.35 Bokmål
UFeats points had been recorded before measurement; it was not a benchmark
claim and was not reached. The failed joint Bokmål/Nynorsk and Rare/OOV gate
does not justify a full retraining.

The frozen adaptive fusion probe was implemented without changing the
selected production checkpoint. It consumed the exact pre-fusion independent
logits, marginalized bundle probabilities, candidate scores, and morphology
token representation from the production forward path. It applied the
selected checkpoint-derived logit correction consistently to both
probability paths and trained only a compact schema-driven gate. Development
labels and UDPipe predictions were never training inputs.

The fixed two-standard run is complete and rejects the probe. Bokmål exact
UFeats falls from 96.6372% to 95.6639% (-0.9733 points, 354 additional wrong
bundles); Rare/OOV falls by 3.4920/3.2063 points. Nynorsk falls from 94.0768%
to 93.7280% (-0.3488 points, 109 additional wrong bundles), with Rare/OOV
losses of 0.6686/1.4196 points. Gender alone loses 281 Bokmål and 138 Nynorsk
correct decisions.

The learned mean gates are near one for almost every feature, including
0.9885/0.9893 for Gender/Number. The training-only gate therefore learned to
prefer the bundle expert almost unconditionally. It improves over the
bundle-only path but does not recover the selected static residual fusion,
especially outside frequent training forms. Falling training loss through
epoch 16 is consequently not evidence of Development generalization. This
rejects the current convex probability-fusion design and its training
contract; it does not justify hyperparameter tuning against Development.
The selected production checkpoint and static fusion remain unchanged. After
the result was recorded, the rejected probe CLI, training module, temporary
fusion boundary, and focused tests were removed again. The production surface
therefore carries no abandoned adaptive-fusion path.

### Candidate-inventory coverage decomposition

The read-only morphology-bundle oracle now measures the actual union of Top-K
candidates across all UPOS groups. For each limit it reports covered tokens,
bundles seen in training but pruned by Top-K, and bundles never seen anywhere
in training. Top-32/64/128 and the full inventory are reported for complete,
annotated, Rare, and OOV slices. No checkpoint is loaded and no training is
performed.

| Split | Top-32 coverage | Seen/pruned | Never seen | Top-64/full coverage |
| --- | ---: | ---: | ---: | ---: |
| Bokmål all | 99.2934% | 256 | 1 | 99.9973% |
| Bokmål Rare | 99.6190% | 11 | 1 | 99.9683% |
| Bokmål OOV | 99.6081% | 11 | 0 | 100.0000% |
| Nynorsk all | 96.9664% | 798 | 150 | 99.5200% |
| Nynorsk Rare | 94.7764% | 96 | 29 | 98.7881% |
| Nynorsk OOV | 95.2287% | 96 | 25 | 99.0142% |

The joint training inventory contains 256 distinct bundles and 298
UPOS-bundle pairs. Top-32 keeps 185 pairs; Top-64 keeps 288. Top-64 removes
every pruning miss on both standards, while Top-128 and the full inventory add
no further Development bundle coverage. Open compositional candidate
generation is therefore deferred: unseen combinations are not the dominant
measured coverage loss. The temporary Top-64 training option and the expanded
coverage-reporting surface were removed after the controlled result was
recorded. The production CLI again exposes only `0` and the selected `32`.

### Controlled Top-64 result

The matched Top-32 control and Top-64 candidate used the same data, seed,
architecture, Base Teacher, training policy, epoch-12 selection, and
canonical full-logit-correction evaluation. Only the retained candidate limit
changed. Runtime candidate coverage rose from 98.2180% to 99.7767%, but this
additional coverage did not produce a joint quality gain:

| Split and metric | Top-32 control | Top-64 candidate | Change |
| --- | ---: | ---: | ---: |
| Bokmål UPOS F1 | 98.9607% | 98.9607% | 0.0000 pp |
| Bokmål UFeats F1 | **96.7005%** | 96.6592% | -0.0413 pp / -15 correct |
| Bokmål Lemmas F1 | **98.9029%** | 98.8919% | -0.0110 pp / -4 correct |
| Bokmål Rare UFeats F1 | **91.0794%** | 90.7937% | -0.2857 pp / -9 correct |
| Bokmål OOV UFeats F1 | **87.2818%** | 87.1749% | -0.1069 pp / -3 correct |
| Nynorsk UPOS F1 | **98.6880%** | 98.6688% | -0.0192 pp / -6 correct |
| Nynorsk UFeats F1 | 94.1632% | **94.2464%** | +0.0832 pp / +26 correct |
| Nynorsk Lemmas F1 | 98.5280% | **98.5504%** | +0.0224 pp / +7 correct |
| Nynorsk Rare UFeats F1 | 87.4634% | **87.5052%** | +0.0418 pp / +1 correct |
| Nynorsk OOV UFeats F1 | 83.8328% | **84.0300%** | +0.1972 pp / +5 correct |

Across both complete splits Top-64 gains only 11 exact UFeats decisions while
losing six UPOS decisions. More importantly, it fails the predeclared
two-standard and Rare/OOV non-regression gate: the larger inventory trades
Bokmål quality for a small Nynorsk improvement instead of reliably exploiting
the 1.5587-point coverage increase. The result rejects Top-64 as the shared
model default and points back to final-output supervision/fusion rather than
more closed candidates.

The newly trained Top-32 control is also not promoted over the selected
checkpoint. Relative to the selected Student it gains 23/27 complete UFeats
decisions on Bokmål/Nynorsk but loses 10/9 Lemma decisions, moving opposite
to the recorded lemma-recovery guardrail. Its purpose remains a matched
control for this ablation. The selected production checkpoint stays
`runs/no-student-character-cnn-dkd-bundle32-direct-loss-morphology-gradient-prehead-shared-mlp-w010-e12-weighted/best.pt`.
The temporary Top-64 CLI/test surface and the completed coverage-decomposition
extension were removed after documentation; no rejected runtime path remains.

A separate flat UFeats classifier is not the planned default. The selected
Bundle-32 reranker already provides the same conceptual parallel whole-bundle
path: it scores complete candidates and marginalizes their distribution back
into the individual features, while the direct bundle loss supplies
whole-bundle supervision. A second classifier over the same inventory would
duplicate that responsibility and inherit its closed-inventory weakness.
Improving the existing scorer tests the demonstrated ranking bottleneck more
directly and preserves one calibrated owner for bundle coherence.

Every trained candidate remains subject to the fixed joint gate: canonical
Bokmål and Nynorsk UFeats and every individual feature, Rare/OOV quality, UPOS,
and Lemmas. The immediate recovery target is at least the 12 Bokmål and nine
Nynorsk lemma predictions lost relative to the selected `identity` control
without giving back the shared MLP's 226 complete-bundle gain. UDPipe
predictions must not enter training, candidate construction, loss weighting,
or threshold selection, and both official test splits remain untouched.

## Versioned model-artifact export

The first versioned model artifact is implemented and produced. The export
pipeline lives in `prism.exporting` (fixed-shape padding, ExecuTorch XNNPACK
lowering, typed manifest, fixtures) with the thin CLI
`prism.languages.norwegian.export_artifact`. It loads the frozen benchmark
checkpoint, embeds the canonical morphology-logit correction 0.25 into the
exported graph, captures the complete character-aware tagger with strict
`torch.export` at static shapes (batch 8, 160 subwords, 96 tokens, 32
characters), lowers it to an XNNPACK `.pte`, and writes
`models/prism-no-<version>/` containing `model-xnnpack.pte`, `manifest.json`,
`labels.json` (task schema plus character vocabulary), `vocabulary.json`
(the NorBERT4 fast-tokenizer definition), `fixtures.json`, and `LICENSES/`.
The directory stays out of Git.

Runtime parity is a gate, not a claim: the CLI replays each recorded fixture
batch through the ExecuTorch runtime and fails the export unless decoded
predictions match eager PyTorch exactly and task probabilities agree within
the documented tolerance (default 5e-3). Raw logits are deliberately not the
parity space because the bundle reranker maps saturated candidate
probabilities through `log`/`logit`; numerically irrelevant runtime
differences near the epsilon clamp inflate to logit gaps of about 0.25 on
Definite, Number, Gender, PronType, and VerbForm while probabilities agree to
about 1.5e-3 and decoding is identical. `torch.export` itself is bit-exact
against eager; the measured artifact records
`parity_maximum_probability_difference` 1.48e-3 across both development
fixture batches. Outputs at padded positions are contractually undefined and
excluded.

Fixtures record, per written standard, eight development sentences with all
padded input tensors, full expected logits for UPOS and every morphology
feature, top-8 lemma-rule logits (the full lemma head would dominate the
file), and canonically decoded predictions a native runtime must reproduce.
The produced `prism-no-0.1.0` artifact measures about 88 MiB for the FP32
program; quantization, confidence calibration, the versioned Nynorsk external
treebank policy, and the document-scale performance gate remain open release
requirements.

The export command:

```bash
python -m prism.languages.norwegian.export_artifact \
  --checkpoint runs/no-student-silver-10m-large-labels-w050-e12-weighted/best-development-task-accuracy.pt \
  --artifact-version 0.1.0
```

## Vorläufiger Architektur-Cleanup

Der angekündigte Cleanup ist abgeschlossen. Entfernt wurden der abgelehnte
Agreement-Refiner, die Task-Familien-Adapter, die nicht mehr benötigten
Task-Head-Zwischenarchitekturen, der alte Gradient-Isolations-Alias und das
abgeschlossene Frozen-Head-Probe-Werkzeug einschließlich CLI-, Checkpoint- und
Testpfaden. Nach seiner Messung wurde auch der verworfene adaptive
Fusions-Probe vollständig aus den ausführbaren Pfaden entfernt. Diese
Experimente bleiben in `docs/benchmarks.md` als Auswahlhistorie
nachvollziehbar, sind aber nicht länger Teil der Produktionsoberfläche.

Erhalten bleiben die ausgewählte vollständige Student-Architektur, der
`linear`-Kontrollpfad, `identity` zur Reproduktion der Morphologie-Vorprojektion,
alle drei expliziten Bundle-Gradient-Scopes, der ausgewählte lineare
Bundle-Scorer, der unmittelbar für die geplante Fusion relevante
kompositionelle Scorer sowie die Evaluation-Audits. Alte Checkpoints der
entfernten Forschungsarchitekturen werden nicht mehr geladen; der ausgewählte
Produktionscheckpoint und sein Format-3-Vertrag bleiben unverändert.

## PrismKit (Swift package, native phase)

`swift/PrismKit` starts the native phase with the layers that carry the
language contract: `RuntimeSegmentation` is the native port of
`prism-runtime-segmentation-v1` (line merging, protected boundaries,
missing-space repair, UD tokenization, chunking) with the Norwegian
abbreviation policy, and `PrismArtifact` decodes the manifest and label
schema including `LemmaEditRule.apply` with reference semantics.
`ComputeDevice` (`automatic`/`cpu`/`gpu`) selects among the manifest's
lowered programs — the artifact contract already carries one program per
backend, so the user-facing device choice resolves at load time and
unavailable devices surface a typed error (pre-Apple-Silicon Macs use the
XNNPACK program). Cross-language parity is enforced twice: the Swift test
suite mirrors the Python runtime-segmentation tests one to one, and the
local book-chapter fixture must segment to exactly the Python reference
counts (247 sentences, 3,783 tokens). 11/11 Swift tests passing. Next
steps: the ExecuTorch engine layer (load `model-xnnpack.pte`, execute a
fixture batch, parity against `fixtures.json`), subword tokenization via
swift-transformers, then the C++ mirror over the same layers for the
LexKeep core and Windows/Linux.

### ExecuTorch engine spike (passed)

PrismKit executes the shipped program natively: the `executorch` Swift
package (branch `swiftpm-1.3.1`, matching the Python exporter version)
loads `model-xnnpack.pte`, and a recorded fixture batch runs end to end
with int64/bool input tensors. Two integration findings are captured in
the package: static backends register through global initializers, so
the final binary must force-load the archives (`-all_load` in the test
target; Xcode consumers set `-force_load` per the ExecuTorch docs), and
the outputs verify as calibrated probability distributions with the
schema-derived label count. 12/12 Swift tests passing. Decision for the
subword layer: PrismKit implements the BPE tokenizer natively
(NFKC + split pre-tokenizer per `vocabulary.json`) with the recorded
fixture `input_ids` as the parity oracle; the Hugging Face tokenizer
remains a documented opt-in alternative. Maximum performance at equal
quality is the API's stated top goal for weak target devices.

### Native BPE subword tokenizer (PrismKit)

`SubwordTokenizer` executes the artifact's `vocabulary.json` definition
natively: NFKC normalization, the GPT-style split pre-tokenizer, the
byte-level table, plain BPE with `ignore_merges`, the `<s>` template,
and the word-to-subword alignment (`first_subword_indices` /
`subword_end_indices`) the pooling consumes. Parity is enforced at two
scales: nine crafted cases recorded from the reference Hugging Face
tokenizer (unicode headings, ordinals, guillemets, compounds, e-mail,
URL, emoji) match IDs and alignment exactly, and the untracked
book-chapter oracle requires all 247 sentences / 5,028 subword IDs of
the combined Swift segmentation + BPE pipeline to equal the Python
output — a joint test of both layers where any divergence surfaces as
an ID mismatch. The full chapter tokenizes in ~13 ms. The Hugging Face
runtime remains a documented opt-in alternative reading the same file.
14/14 Swift tests passing. Next: `tagPretokenized` (fixed-shape batch
assembly, character IDs, ExecuTorch execution, argmax/threshold/lemma
decoding), then `tagText`, then the performance pass.

### PrismTagger (Swift end-to-end pipeline)

`PrismTagger` completes the Swift API surface: `tag(text:)` runs the
runtime segmentation (bounded by the program's fixed token count),
`tag(pretokenized:)` accepts application words, and both assemble the
fixed 8×160×96×32 batches natively — repeat-padded partial batches,
subword padding with the manifest's padding ID, zero-padded alignment
indices, and the character encoding with start/end/truncation markers
and NFC lookup mirroring the Python contract. Batches beyond eight
sentences loop transparently; over-long sentences are re-chunked to the
program's capacity. Decoding stays mathematics-free: argmax over the
calibrated probability outputs, the 0.5 threshold for multi-valued
features, and `LemmaEditRule.apply`. End-to-end tests reproduce the
Python reference decisions on raw text (UPOS, features, lemmas,
confidence ranges) and across the multi-batch path. 16/16 Swift tests
passing. Remaining for the Swift phase: the performance pass (scanner-
based segmentation and buffer reuse are the known levers) and the
README integration notes.

### Performance pass (multi-shape programs + length-sorted bucketing)

The release-mode baseline showed PrismKit's own code costs ~11 ms of the
5,645 ms chapter run — the fixed 8×160×96 shapes dominate, padding every
~20-subword sentence to 160 subwords. The lever is therefore shape
bucketing, not micro-optimization: the export gained `--small-shapes`
(a second lowered program, here 8×48×32, parity-gated exactly like the
main program: max |Δp| 1.63e-3, decoded predictions identical), the
manifest lists both programs, and `PrismTagger` sorts sentences by
subword length, batches them, and runs every batch on the smallest
program it fits into (modules load lazily; results scatter back to
input order). Chapter benchmark: **5,645 ms → 2,949 ms** (670 → 1,283
tokens/s), 17/17 Swift tests passing including the reference-decision
tests on the bucketed path. Trade-off recorded honestly: each program
embeds the full weights, so the two-program artifact is ~177 MB on
disk; ExecuTorch program-data separation (one shared weight file,
small per-shape programs) is the documented follow-up if disk size
matters for distribution. Further bucket sizes remain available via
repeated exports when profiling justifies them.

## C++ phase (started)

`cpp/` begins the C++ mirror of the native layers, styled after the
LexKeep core it will integrate with (codepoint spans, table-based
character classes, hand-written scanners — no regex engine, no ICU).
`prism::segmentation` ports `prism-runtime-segmentation-v1` completely:
UTF-8 decoding, missing-space repair, wrapped-line merging with
dehyphenation, protected sentence boundaries, the UD token scanner
(URL, e-mail, number, dotted-abbreviation, word, fallback — in the
reference pattern's alternative order), period reattachment, spacing,
and chunking. Letter classification covers Latin (all Norwegian
characters), Greek, and Cyrillic; other scripts tokenize character-wise
as documented. The test binary mirrors the Python and Swift suites and
enforces the book-chapter reference counts: 247 sentences / 3,783
tokens, segmented in **1.67 ms** (Swift: 5.4 ms, Python: 9 ms). CMake
with ctest; all checks passing. Remaining C++ roadmap: the BPE subword
tokenizer port with the shared parity fixtures, artifact/label parsing,
the ExecuTorch C++ runtime engine, and the C ABI for the LexKeep core
and the Windows/Linux variants.

### C++ BPE subword tokenizer

`prism::subword::Tokenizer` ports the byte-level BPE natively: the
GPT-style split pre-tokenizer as a scanner, the byte-to-unicode table,
whole-piece lookup with `ignore_merges`, ranked merges, the `<s>`
template, and the word alignment. Diagnosing the first parity run
surfaced that the reference pipeline's NFKC folds characters like
U+2026 into ASCII ("…" → "..."); the port applies a compact
compatibility table for the practically occurring subset (ellipses,
ligatures, no-break and typographic spaces, №, ™) with the shared
fixtures as the gate and pass-through for anything else. Parity: the
nine recorded cases (IDs and alignment) and all 247 chapter sentences /
5,028 subword IDs match the reference; segmentation plus BPE process
the chapter in ~3.6 ms. JSON parsing uses nlohmann/json via pinned
FetchContent. Both ctest suites passing. The Xcode workspace issue was
fixed separately: the swiftpm-1.3.1 frameworks were built with Swift
5.10, the current swiftpm-1.4.0 snapshot compiles from its textual
interface, and the 1.3.1-exported program runs unchanged on the 1.4
runtime (verified by the engine tests and xcodebuild).

### C++ ExecuTorch engine

`prism::engine::Program` executes the shipped `.pte` programs from C++
behind a PIMPL boundary, so ExecuTorch headers never leak into Prism's
API — the shape the later C ABI needs. The runtime cannot be vendored
as an amalgamation (the source tree with submodules spans hundreds of
megabytes); instead it is pinned to the exporter's exact version
(FetchContent, tag v1.3.1) behind `option(PRISM_ENGINE ON)`, so
checkouts without network still build everything else. Source-build
findings captured in CMake: the ExecuTorch preset chain requires
`EXTENSION_NAMED_DATA_MAP`, the source directory must be named exactly
`executorch`, torch headers resolve through the Python interpreter
(the repository venv is wired as the default), and kernel/backend
registration needs the whole-archive helper. The engine test executes a
recorded fixture batch on `model-xnnpack.pte` and compares against the
recorded expectations: **max |Δ| = 1.8e-06** — the same backend as the
recording, so effectively exact. 10/10 GoogleTest tests passing.
Remaining C++ roadmap: artifact/label parsing, the tagger with
multi-shape bucketing, and the C ABI for the LexKeep core.

### C++ tagger and C ABI

The C++ API is now complete end to end. `prism::artifact::Artifact`
parses the manifest and the label schema (programs sorted by capacity,
tokenizer contract, morphology features, codepoint-based lemma edit
rules, character vocabulary); `prism::tagger::Tagger` mirrors the full
runtime pipeline: runtime segmentation bound to the largest program's
token capacity, native BPE encoding, character encoding with the
reserved ids and middle truncation, length-sorted batching across the
fixed-shape programs (smallest fitting program per batch, lazily loaded
engines, repeat-padded partial batches), and decoding of the calibrated
probabilities (argmax, NONE-first exclusive features, 0.5 threshold
with minimum confidence for multi-valued features, lemma rules with
token fallback). On top sits the C ABI (`prism/prism_c.h`): opaque
`prism_tagger`/`prism_result` handles, thread-local `prism_last_error`,
raw-text and pretokenized entry points, and accessors exposing only C
types — tokens, UPOS, per-feature iteration plus a CoNLL-U style
feature string, lemmas, and all calibrated confidences. This is the
surface the LexKeep core (and the Windows/Linux variants) will link.
Verification: the tagger reproduces the recorded reference decisions
("Hun kjøpte tre gamle bøker den 17. mai." with lemmas and morphology),
the C ABI test exercises every accessor including error paths, and the
book chapter tags end to end in ~4.0 s (247 sentences / 3,783 tokens,
matching the reference counts). 15/15 GoogleTest tests passing.
Swift-specific wording was removed from the C++ documentation — the
C++ binding stands on its own. Follow-up lever: the chapter runs ~35%
slower than the Swift runtime on the same programs; worth profiling
the XNNPACK threadpool configuration before optimizing own code.

### Umbrella header and Java API — the platform set is complete

Two additions round off the bindings. First, `#include <prism>` now
pulls in the complete C++ API: an extensionless umbrella header in its
own include directory (`cpp/umbrella/`, since a file cannot share the
name of the `include/prism/` directory) plus an aggregate CMake
INTERFACE target `prism`, so consumers link one target and get every
library and both include paths. Second, `java/` adds a dependency-free
Java 21 API (`io.github.dmlux.prism`: `PrismTagger` as `AutoCloseable`,
`TaggedSentence`/`TaggedToken` records with features and calibrated
confidences) over a JNI bridge (`cpp/src/jni.cpp`) onto the C++ tagger.
Design decisions: text crosses the boundary as UTF-8 byte arrays and
results return as one flat parallel-array payload per call (a single
JNI transition regardless of sentence count); returned strings use
`NewString` with a hand-written UTF-8→UTF-16 conversion because JNI's
modified UTF-8 breaks for supplementary codepoints. The canonical build
stays CMake-only (`PRISM_JAVA`, on by default when a JDK 21 and JNI are
found; UseJava builds `prism.jar`), while `java/pom.xml` (standard
Maven layout, zero dependencies, coordinates `io.github.dmlux:prism`)
serves Maven/Gradle consumers — Maven itself is not installed locally,
so the POM is unverified. The Java test is a plain `main` program (no
JUnit dependency) run through ctest with `java.library.path` pointed at
the build tree; it validates the recorded reference decisions, the
multi-batch path, and the error path. ctest: 2/2 suites (16 GoogleTest
tests + Java) passing. With Swift, C++/C, and Java/Kotlin the platform
set from the roadmap is covered; packaging levers (natives inside the
JAR per platform, published Maven artifact) remain deliberate follow-ups.

### Program-data separation: artifact 0.2.1

The two-program artifact carried every weight twice (2 × 88 MB), which
broke the 100 MB app-bundle premise — the constraint is shipping size,
not runtime memory, and it compounds with every added language. Artifact
`models/prism-no-0.2.1` fixes this with ExecuTorch program-data
separation: the weights live exactly once in `model.ptd` (83.3 MiB) and
both fixed-shape programs shrink to graph-only files of **0.7 MiB each**
(shipping bundle with both shapes ≈ 92 MB, premise restored). Two export
passes are required to cover every weight: the delegate pass tags
constants consumed by the XNNPACK payload, and the backend-config
callable tags what the delegate leaves in the program — most importantly
the subword embedding, which alone dominated the program size (the first
attempt externalized only 2.6 MiB because it used the delegate pass
alone). Programs reference tensors by content hash, so the small program
executing against the main program's data file is itself the gate that
the weights are byte-identical across shapes. The manifest grows
`data_files` per program plus a checksummed top-level `data_files`
provenance list; the exporter gains `--external-data`. All runtimes load
the listed data files alongside the program: Swift through
`Module(filePath:dataFilePaths:)`, C++ through the module data-files
constructor (surfaced in `prism::engine::Program` and resolved from the
manifest by the tagger), Java via the C++ path, and the Python parity
gates through the pybindings' `data_path`. Full verification against
0.2.1: Python 302/302, Swift 17/17, C++/Java ctest 2/2 suites (16
GoogleTest tests + the Java end-to-end run), chapter timings unchanged.
For the many-language outlook the remaining levers stay documented:
int8 XNNPACK quantization (~22 MB per language) and downloading language
artifacts on demand instead of bundling them.

### First public model release: prism-no 0.2.1

Published as a GitHub release (tag `prism-no-0.2.1`, assets:
`prism-no-0.2.1.tar.gz` with the complete artifact directory,
`SHA256SUMS`, and a loose `manifest.json` for inspection without
downloading). Release conventions decided here: model releases are
tagged per language and version (`prism-{language}-{semver}`) so every
language keeps an independent lifecycle; plain `v*` tags stay reserved
for future library releases; GitHub Releases carry the artifacts (free
storage and bandwidth, no LFS cost trap), with a Hugging Face mirror as
documented follow-up. Licensing decided and recorded in the release
notes: source code Apache 2.0, model weights CC BY-SA 4.0 — matching
the gold annotations' share-alike terms by construction; using or
bundling the unmodified artifact (including commercially, closed
source) is not an adaptation, while redistributed modified weights must
stay open. The notes credit every training-data source with license and
revision: the UD Bokmål/Nynorsk treebanks (CC BY-SA 4.0), Språkbanken's
NBdigital `sbr-43` and municipal-documents `sbr-60` corpora (CC0,
National Library of Norway), the Nynorsk Wikipedia dump (CC BY-SA 4.0,
text never redistributed), and the `ltg/norbert4-xsmall` backbone
(Apache 2.0, University of Oslo). The release states explicitly that
one set of weights serves both written standards and that the artifact
ships CPU (XNNPACK) programs as a measured decision, with the device
API ready for future GPU-lowered programs.

### Runtime optimization: thread cap and four-shape artifact 0.2.2

Two measured levers close most of the gap to the eager-Python
throughput target. (1) The ExecuTorch threadpool default spans every
logical core because cpuinfo does not separate performance from
efficiency cores on Apple Silicon; a sweep on the 16-core M4 Max showed
6 threads beating the default by 24% on the small fixed-shape batches.
`prism::engine` now exposes `ThreadCount`/`SetThreadCount` (plus
`SetDefaultThreadCount`, which never overrides an explicit choice), the
C ABI gains `prism_set_thread_count`, Java `PrismTagger.setThreadCount`,
and the C++ tagger installs the measured default of 6. (2) Bucketing
analysis on the chapter showed 150 of 247 sentences fit 8×24×16 while
everything ran on 48×32 or larger; the exporter now accepts repeated
`--small-shapes`, and artifact 0.2.2 ships four programs (24×16, 48×32,
96×64, 160×96 — each +0.7 MB thanks to the shared model.ptd, bundle
still ≈ 94 MB). Chapter results (warm): C++ 3.9 → 2.2 s, Java
3.9 → 2.2 s, Swift 2.9 → 2.5 s (Swift cannot cap threads through the
prebuilt frameworks yet — upstream follow-up). C++/Java now lead the
native field; eager Python remains at 1.6 s as the dynamic-shape
bound. All suites verified against 0.2.2: Python 302/302, Swift 17/17,
C++/Java ctest 2/2; every added program passes the parity gate against
the shared data file.
