# Engineering guide

Source code defines reusable behavior; YAML defines a complete experiment.
Keep data access, transformations, model structure, training, and evaluation at
separate boundaries so a change can be reviewed and tested in isolation.

## Responsibilities

| Module | Owns | Does not own |
|---|---|---|
| `config.py` | YAML parsing and experiment configuration validation | Model fitting or data access |
| `data.py` | Parquet loading, schema validation, consumer splitting | Feature formulas or model calls |
| `features/baseline.py` | Cashflow, category, and income-gap aggregates | File access or fitting |
| `features/temporal.py` | Monthly trend and frequency-domain transformations | File access or fitting |
| `features/risk.py` | Behavioral-risk transformations | File access or fitting |
| `features/balance.py` | Reconstructed-balance features | File access or fitting |
| `features/sets.py` | Named, ordered feature selections | Training policy |
| `features/pool.py` | Compose feature families from input frames | Reading parquet or fitting models |
| `models/` | Model state, component construction, and prediction | Dataset loading or resampling |
| `sampling.py` | Training-only resampling primitives | Model architecture or persistence |
| `trainer.py` | Fit the router and per-group experts | Feature formulas or file loading |
| `evaluations/` | Metrics and repeated permutation importance | Training or feature selection policy |
| `experiment.py` | Wire components and persist a run | Feature or sampling algorithms |
| `main.py` | Parse CLI, load configuration, initialize runtime | Experiment business logic |
| `demo.py` | Generate synthetic fixtures and call the same experiment pipeline | A separate training implementation |

## Make one experiment change at a time

1. Copy the closest configuration and give the experiment a descriptive name.
2. Change the feature bundle, sampling policy, or model settings being studied.
3. Keep seed, data, and evaluation protocol fixed unless they are the subject of
   the comparison. Check the saved split assignments when comparing runs.
4. Run `uv run cashflow-run --config configs/<experiment>.yaml`.
5. Compare the saved metrics and per-client predictions. Use validation for
   iteration and feature selection; use test for the final evaluation.

All hyperparameters, data locations, thread counts, sampling choices, and
permutation settings belong in YAML. New features should be stateless functions
over supplied frames and registered in the feature catalog. Keep historical
aliases when renaming bundles, so existing experiment configurations remain
readable and runnable.

## Environment and artifacts

Use `uv add <package>` for dependencies and commit both `pyproject.toml` and
`uv.lock`. Use `uv sync` to recreate the environment. New code must not install
packages or mutate thread settings at import time; entry points initialize
runtime settings before loading numerical libraries.

Every experiment gets a timestamped `logs/` directory containing its config,
metrics, run log, feature order, split assignments, input/environment metadata,
predictions, and model. `config.yaml` records the effective configuration used
by the run, including programmatic changes; `source_config.yaml` preserves the
original supplied file. Optional analyses live beside those outputs. Data and
artifacts never go under `src/` and are excluded from git. Input metadata records
file size and modification time; retain the exact source data separately when
full dataset versioning is required.

## Checks that matter

```bash
uv run pytest
uv run ruff check .
uv run cashflow-demo --config configs/demo.yaml
```

Tests should cover behavior: preserved feature values and predictions, split
disjointness, observation cutoffs, correct sampling and weighting, and artifacts
from the real experiment path. Synthetic data provides reproducible checks
without distributing source records. A refactor and an algorithm correction are
different changes: preserve numerical behavior during reorganization, then
evaluate any modeling correction through a separate configuration and run.
