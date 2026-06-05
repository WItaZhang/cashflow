# Machine Learning Project Guide

## Principles

Keep experiments reproducible, boring, and inspectable. Source code defines
reusable behavior; YAML config defines a specific run. Never create a new
`evaluation_v17.py` just to change a feature list, model version, or metric.

## Recommended Structure

```text
configs/                 one YAML per experiment
data/raw/                read-only source data or symlinks
data/processed/          reusable processed feature artifacts
data/staging/            temporary caches and split files
logs/YYYYMMDD_HHMMSS_*   run outputs
artifacts/               exported submissions or reports
src/cashflow_ml/
  data.py                loading and splitting
  features/              feature builders, feature sets, feature pool
  models/                model classes and model registry
  evaluations/           metrics and evaluation registry
  experiment.py          run orchestration
  main.py                CLI entrypoint
```

## Experiment Workflow

1. Add or edit feature builders in `src/cashflow_ml/features/`.
2. Register readable feature names in `src/cashflow_ml/features/sets.py`.
3. Add new model structures under `src/cashflow_ml/models/` and register them.
4. Add metrics under `src/cashflow_ml/evaluations/registry.py`.
5. Create a new YAML file in `configs/`.
6. Run `uv run cashflow-run --config configs/<experiment>.yaml`.
7. Compare `metrics.json`, `test_predictions.csv`, and `model.joblib` under the run folder.

## Versioning Rules

- Config file name is the experiment identity.
- `uv.lock` is committed and is the environment source of truth.
- Every run stores a copy of the exact config used.
- Model files are stored under the run folder, not under `src/`.
- Predictions and metrics are stored with the run, not overwritten globally.

## Feature Pool Rules

Features should be selected by readable names, not copied lists inside scripts.
Use coarse sets for common bundles and small named sets for ablations:

```yaml
features:
  feature_sets:
    - v15_baseline_44
    - v15_4_fft_6
    - v15_4_risk_6
```

When a feature is removed because of selection, keep the reason in the set name
or in a short comment near the set definition.

## What To Avoid

- One-off scripts named by version that duplicate most of the project.
- Hardcoded data paths or hyperparameters in source files.
- Writing logs, predictions, or model weights into `src/`.
- Changing multiple dimensions at once without a config name that says so.
- Keeping feature selection results only in notebooks or terminal history.
