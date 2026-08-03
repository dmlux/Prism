# Prism repository guidance

## Start here

- Before planning or changing the project, read `README.md`,
  `docs/PROJECT_STATUS.md`, and `docs/BENCHMARKS.md`.
- Treat `docs/PROJECT_STATUS.md` as the handoff for the confirmed current state
  and update it whenever a milestone, benchmark, or architectural decision
  changes materially.
- Read `docs/MODEL_STRATEGY.md` before changing model architecture, task heads,
  export formats, native runtimes, or the Python package layout.
- Verify the repository instead of assuming that an older handoff is still
  current.

## Collaboration and teaching

- Communicate with the user in German unless they request another language.
- The user is learning machine learning and Python project development. Explain
  new concepts in plain language without assuming prior ML experience.
- For the guided learning workflow, provide exactly one concrete next step,
  briefly explain what it does and why it is needed, then wait for the user's
  result before continuing.
- When the user asks Codex to implement, inspect, fix, or verify something
  directly, perform that bounded task and report the outcome; do not turn it
  into a long homework list.
- Explain what is being planned and why before introducing a new model,
  training objective, dataset representation, or evaluation metric.

## Project direction

- Prism is a modular, open-source NLP toolkit with local, privacy-friendly
  inference. Norwegian Bokmål is the only current language target.
- Design the architecture for many languages from the start even though
  Norwegian Bokmål is the only current implementation target. Keep the public
  data, training, inference, evaluation, export, and native API contracts
  language-independent.
- Put every replaceable language decision behind an explicit language profile
  or similarly narrow interface. This includes teacher and student backbones,
  tokenizer behavior, normalization, dataset adapters, morphology and lemma
  schemas, label inventories, decoding, artifact metadata, and licenses.
- Treat NorBERT4 as one Norwegian backbone configuration, never as an
  assumption of the generic Prism model pipeline. Generic batching,
  subword-to-token alignment, task-head construction, losses, distillation,
  calibration, evaluation, and export code must not import or branch on a
  concrete language model.
- Reuse the same UPOS, per-feature morphology, lemma, and confidence head
  implementations across languages. Their output dimensions and labels come
  from the selected language artifact schema; do not hard-code the Norwegian
  feature inventory or label counts into shared heads.
- Preserve a path toward native Swift packages and embedding into host
  applications. Host applications often own tokenization and source offsets
  already, so the APIs support externally supplied tokens as well as the
  high-level raw-text pipeline.
- Do not silently combine token tagging with phrase, named-entity, multiword-
  expression, tokenization, or sentence-segmentation work. Those are distinct
  tasks and require an explicit design decision.
- Accuracy and calibrated uncertainty matter because predictions may be shown
  in learning software. Do not present a single aggregate accuracy as proof
  that every class or prediction is dependable.
- Follow the accepted teacher-student direction in `docs/MODEL_STRATEGY.md`:
  use a high-capacity Norwegian teacher for quality-oriented training and ship
  only a compact, measured student for local inference.
- The first new production model bundle covers UPOS, the supported Norwegian
  UD morphology features, lemmatization, and calibrated confidence. Keep
  dependency parsing, tokenization, sentence segmentation, named entities,
  phrases, and multiword expressions as explicit later decisions.
- Treat the versioned model artifact and its manifest as the cross-platform
  contract. Keep public Swift, Java/Kotlin, and C++ APIs independent of the
  selected inference runtime.

## Python and package conventions

- Use Python 3.12 and the repository-local `.venv`.
- Install development dependencies with
  `python -m pip install -e './python[dev]'`.
- The distribution name is `prism-nlp`; the Python import package is `prism`.
- Run modules from the repository root with `python -m prism.<module>`.
- Keep reusable behavior in modules under `python/src/prism/`. Short terminal
  snippets are acceptable for exploration, but repeatable training,
  evaluation, and prediction workflows belong in Python modules and tests.
- Prefer type annotations and small, explicit data transformations that are
  understandable to an ML beginner.

## Python architecture quality

- Treat Python as production code, not as a collection of experiment scripts.
  Before adding a new module, identify its responsibility, its public contract,
  and the existing layer that should own it.
- Organize reusable behavior by stable concerns such as data/schema, task
  definitions, model components, training, evaluation, export, and artifact
  loading. Keep dependencies directed: lower-level schema and task contracts
  must not import CLI, training orchestration, or a specific native runtime.
- Make generic model code depend on typed backbone and language-profile
  contracts rather than concrete NorBERT, Norwegian, or future language
  implementations. Language packages may depend on the generic core; the
  generic core must not depend on a language package.
- Keep `python -m prism.<module>` entry points thin. They may parse arguments,
  construct configuration, call reusable services, and render results; model,
  data, training, evaluation, and checkpoint logic belongs in importable
  modules with tests.
- Use explicit typed domain objects for configurations, batches, predictions,
  metrics, manifests, and checkpoint metadata instead of loosely structured
  dictionaries crossing module boundaries. Validate data at file, checkpoint,
  and model-artifact boundaries and fail with actionable errors.
- Separate pure transformations from filesystem access, device selection,
  logging, and command-line output. Pass dependencies and configuration
  explicitly rather than relying on mutable module-level state or hidden
  defaults.
- Extend or refactor the existing source of truth instead of copying logic into
  another training, evaluation, or prediction path. Shared behavior must have
  one tested implementation; task-specific differences should be represented
  through clear configuration or narrow interfaces.
- Prefer small cohesive modules and composition over large classes, inheritance
  trees, and catch-all utility files. Introduce an abstraction only when it
  expresses a real stable boundary or removes demonstrated duplication; do not
  add design patterns merely to make the project appear sophisticated.
- Keep model architecture, training policy, evaluation policy, serialization,
  and runtime integration separable. A change to one should not require
  rewriting the others unless their documented contract genuinely changes.
- Make reproducibility data first-class: resolved configuration, random seeds,
  dataset identity, label schema, model schema version, and relevant library
  versions must travel with checkpoints and released artifacts.
- Add tests at the layer where behavior belongs. Prefer focused unit tests for
  transformations and contracts, integration tests for checkpoint/export
  boundaries, and a small number of representative end-to-end commands.
- Do not append new production code to a weak structure merely to move faster.
  Improve the affected boundary first when necessary, preserve behavior with
  tests, and then add the feature. Keep such refactors scoped and preserve
  unrelated user work.

## Data, training, and evaluation

- The current dataset is Universal Dependencies Norwegian Bokmål at commit
  `396d11f0c2bd290a2a2711015c04ac25bc3dcc06` under CC BY-SA 4.0.
- Keep downloaded datasets, checkpoints, virtual environments, caches, and
  generated training artifacts out of Git.
- Do not change the pinned dataset revision or split definitions without
  documenting the decision and creating a new comparable benchmark.
- Use the training split to fit parameters, the development split to select
  models and tune decisions, and the test split only for final evaluation of a
  fixed model.
- Keep random seeds and relevant hyperparameters in checkpoints and benchmark
  documentation.
- Report morphology accuracy both overall and on annotated tokens because the
  `<NONE>` class otherwise inflates the headline score.
- Preserve existing checkpoint compatibility when practical. If a format must
  change, document the migration or explicitly version the new format.
- Treat the selected class-weighted Transformer student as the gold-only
  reference for teacher-distillation ablations. Do not reintroduce the removed
  recurrent or dictionary experiment paths.
- For teacher-student work, report an ablation against the same student trained
  without distillation. A larger teacher is not evidence that the shipped
  student improved.

## Verification

- After Python changes, run `python -m pytest python/tests`.
- For changes to inference, checkpoint loading, or model structure, also run
  the relevant language-specific training or evaluation smoke command when
  its local data and checkpoint are available.
- For model export, verify numerical parity between PyTorch and the exported
  artifact. For production-runtime changes, measure the 6,000-token,
  200-sentence document fixture using the protocol in
  `docs/MODEL_STRATEGY.md`.
- Check `git diff --check` before handing off code changes.
- Never claim a benchmark improved without evaluating the fixed checkpoint on
  the appropriate split and recording the exact result.

## Licensing and repository hygiene

- Prism source code is Apache License 2.0. External datasets and model artifacts
  retain their own licensing and attribution requirements.
- Do not commit UD data or trained model files. A model release must document
  dataset provenance, dataset license, configuration, metrics, and model
  licensing separately.
- Preserve unrelated user changes in a dirty working tree.
- Do not commit, push, rename remotes, or publish model artifacts unless the
  user explicitly asks.
