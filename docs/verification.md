# Refactor verification

The reference implementation is commit
[`d23acc4`](https://github.com/WItaZhang/cashflow-credit-risk/tree/d23acc43cb4930a8984d8c74e9cdb2b871fcc056).
These checks validate implementation equivalence, not credit-risk performance.

## Numerical comparison

The original source was archived and executed separately in the same locked
Python 3.10 environment as the refactored implementation. Both versions received
the same deterministic synthetic parquet inputs: 180 consumers in three client
groups and 10,800 transactions. Original model hyperparameters and seeds were
retained.

| Original configuration | Model inputs | Features | Split membership | Validation/test predictions |
|---|---:|---|---|---|
| `v15_4.yaml` | 56 | Exactly equal | Exactly equal | Exactly equal |
| `v15_4_balance.yaml` | 60 | Exactly equal | Exactly equal | Exactly equal |
| `v15_4_balance_mean.yaml` | 57 | Exactly equal | Exactly equal | Exactly equal |

Feature comparisons included row order, column order, dtypes, and missing values.
The maximum absolute prediction difference was `0.0` in all three comparisons.
A separate training comparison used 600 synthetic rows, 56 features, missing
values, and both weighted and unweighted policies; all predictions were also
exactly equal under the original full LightGBM hyperparameters.

The feature table now consistently retains `evaluation_date`. Previously an
empty historical aggregation could drop this metadata column through a merge
collision. It is not a model input and restoring it did not change predictions.

## Repeatable project checks

```bash
uv sync --locked
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python -m pytest -q
uv run --frozen cashflow-demo --config configs/demo.yaml
```

The test suite covers information cutoffs, feature definitions, split isolation,
group sampling, historical weights, soft-routing class alignment, model artifact
round trips, permutation importance, configuration snapshots, logs, and a full
synthetic run. GitHub Actions executes these checks on Windows and Linux.

Behavioral hardening is limited to explicit errors for malformed inputs,
unusable experts, invalid routing probabilities, and undefined holdout AUC;
removing the silent non-LightGBM fallback; restoring metadata; and retaining the
effective runtime configuration alongside its source YAML. Original sampling
weights and observed-month feature semantics remain documented in
[methodology](methodology.md).
