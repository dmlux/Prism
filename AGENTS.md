# Prism repository guidance

## Start here

- Before planning or changing the project, read `README.md`,
  `docs/PROJECT_STATUS.md`, and `docs/benchmarks.md`.
- Treat `docs/PROJECT_STATUS.md` as the handoff for the confirmed current state
  and update it whenever a milestone, benchmark, or architectural decision
  changes materially.
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
- Keep language-specific models separable behind a future unified API.
- Preserve a path toward native Swift packages and integration into LexKeep.
  LexKeep already owns tokenization and source offsets, so future APIs should
  support externally supplied tokens as well as an eventual high-level raw-text
  pipeline.
- Do not silently combine token tagging with phrase, named-entity, multiword-
  expression, tokenization, or sentence-segmentation work. Those are distinct
  tasks and require an explicit design decision.
- Accuracy and calibrated uncertainty matter because predictions may be shown
  in learning software. Do not present a single aggregate accuracy as proof
  that every class or prediction is dependable.

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

## Verification

- After Python changes, run `python -m pytest python/tests`.
- For changes to inference, checkpoint loading, or model structure, also run a
  representative `python -m prism.evaluate_*` or `python -m prism.predict_*`
  command when the required local data and checkpoint are available.
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
