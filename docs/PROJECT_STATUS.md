# Prism project status

Last updated: 2026-07-23

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

On the shared development data, Prism leads UDPipe slightly on UPOS for both
written standards and trails slightly on lemmas. Complete-bundle UFeats is the
clear remaining gap: 4.1547 percentage points on Bokmål and 4.1376 points on
Nynorsk. Full numbers and commands are recorded in `docs/benchmarks.md`. The
official test splits remain unevaluated.

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

That gradient-isolation ablation is implemented. The CLI flag
`--isolate-morphology-bundle-loss-gradient` gives the direct auxiliary loss
numerically identical candidate scores, but autograd can update only the
bundle reranker's token-to-candidate projection. The evidence derived from
UPOS and independent morphology logits, the input token representation,
Backbone, shared MLP, task heads, and refinement gates are detached for this
term. Normal supervised losses, distillation, the forward decoder, checkpoint
parameters, inference, and export remain unchanged. The resolved boolean is
stored under `training_config` in new checkpoints. The option requires a
positive bundle-loss weight and is disabled by default, preserving old runs.
Focused tests verify both score equality and the exact gradient boundary.

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
