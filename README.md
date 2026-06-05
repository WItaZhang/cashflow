# Cashflow ML

Config-driven machine learning experiments for the cashflow competition. The
project separates data loading, feature construction, model definitions,
evaluation, and run orchestration so experiments are reproducible and easy to
compare.

## What This Project Does

- Builds tabular features from consumer and transaction parquet data.
- Selects features through named feature sets in configuration files.
- Trains a soft-routing LightGBM mixture-of-experts model.
- Evaluates group-level AUC on per-client train/validation/test splits.
- Saves every run into an isolated timestamped log directory.

## Repository Layout

```text
configs/                  Experiment YAML files
data/                     Local data only; contents are not committed
docs/                     Project notes and feature documentation
logs/                     Timestamped experiment outputs; ignored by git
artifacts/                Exported reports or submissions; ignored by git
src/cashflow_ml/
  data.py                 Data path validation and train/validation/test split
  experiment.py           End-to-end experiment orchestration
  features/               Feature builders, feature pool, and named feature sets
  models/                 Model registry and LightGBM MoE implementation
  evaluations/            Metrics and evaluation registry
  main.py                 CLI entry point
```

## Data Policy

Raw data, processed data, run outputs, model files, predictions, and exported
artifacts are intentionally excluded from git. Keep local parquet files under
`data/` or point the YAML config to another local data location.

The tracked `data/`, `logs/`, and `artifacts/` directories contain only
`.gitkeep` placeholders so the expected structure is visible after cloning.

## Setup

This project uses `uv`; `uv.lock` is the source of truth for the environment.

```bash
uv sync
```

## Configure An Experiment

Each run is defined by a YAML file in `configs/`. Update the `data` section to
match the location of your local files:

```yaml
data:
  consumer_file: data/raw/consumer_data.parquet
  transactions_dir: data/raw/transactions
```

Useful configuration sections:

- `features.feature_sets`: named feature bundles such as `v15_4_keep_56`
- `model.name` and `model.version`: model registry entry
- `model.hyperparameters`: router and expert parameters
- `evaluation.name`: evaluation registry entry
- `runtime.n_jobs`: thread count for training

## Run

```bash
uv run cashflow-run --config configs/v15_4.yaml
```

The same entry point can also be run as a module:

```bash
uv run python -m cashflow_ml.main --config configs/v15_4.yaml
```

Each run creates a folder like:

```text
logs/YYYYMMDD_HHMMSS_<experiment_name>/
  config.yaml
  run.log
  metrics.json
  models/model.joblib
  predictions/test_predictions.csv
```

## Current Baseline

`configs/v15_4.yaml` defines the main baseline:

- 44 baseline cashflow and transaction features
- 6 FFT-derived temporal features
- 6 risk and transaction-behavior features
- Per-client stratified 60/20/20 train/validation/test split
- Soft-routing LightGBM mixture of experts for C01, C02, and C03 clients
- Negative downsampling and positive SMOTE-style augmentation for selected
  client groups

## Development Notes

- Add feature logic under `src/cashflow_ml/features/`.
- Register feature bundles in `src/cashflow_ml/features/sets.py`.
- Add model implementations under `src/cashflow_ml/models/` and register them
  in `src/cashflow_ml/models/registry.py`.
- Add or update metrics under `src/cashflow_ml/evaluations/`.
- Create a new YAML file for each experiment instead of hardcoding experiment
  choices in source code.

## Reproducibility

Every experiment run stores a copy of the exact config used, a run log, metrics,
model artifacts, and predictions in a dedicated timestamped directory. This
makes it possible to compare experiments without overwriting previous results.
