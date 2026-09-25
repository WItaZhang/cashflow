# Cashflow Credit Risk

Credit-risk modeling across three client groups with a **soft-routing LightGBM
mixture of experts**. Transaction histories and consumer balances become 56
features; a learned router combines three group-specific risk predictions.

![Transaction history and consumer balance become a shared 56-feature vector. A LightGBM router weights the predictions of three group-specific experts to produce a credit-risk score.](docs/figures/architecture.png)

*Model overview. The router and all three experts receive the same feature vector.
Each expert predicts risk; the router learns how much weight to assign to each
prediction. Client labels supervise router training; inference uses features alone.*
[Editable SVG](docs/figures/architecture.svg) ·
[Vector PDF](docs/figures/cashflow-methods.pdf)

The original consumer and transaction data are private. The public repository
includes the implementation, experiment configurations, regression checks, and a
synthetic demo. The diagram illustrates the computation without using consumer data.

## The approach

- **Soft-routing mixture of experts.** One multiclass LightGBM router learns
  client-group membership from features. Three binary LightGBM experts learn
  within their respective groups; router probabilities weight their predictions.
- **56 cashflow features.** 44 cashflow and trend features, 6 spectral features,
  and 6 behavioral-risk features. The feature catalog is explicit and ordered.
- **Group-specific sampling.** Selected groups use negative downsampling and
  nearest-neighbor interpolation of positive examples. Only the training split
  is resampled; the router sees the original training population.
- **Inspectable evaluation.** Per-client and mean client ROC-AUC, saved split
  assignments and predictions, and optional repeated permutation importance.

The design allows groups to have different decision functions while blending
their predictions for each consumer. [Methodology and preserved behavior](docs/methodology.md)
describe routing, sampling, and evaluation. Model comparisons use matched splits
and training policies.

## Run the demo

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync
uv run cashflow-demo --config configs/demo.yaml
```

This creates 260 synthetic consumers and 24,960 transactions under
`data/staging/demo/`, trains the same model architecture with a small tree budget,
and computes validation permutation importance for three example features.
Generation, feature construction, splitting, training, and evaluation use the
configured seed. Identical existing demo inputs are reused; different existing
data is never overwritten.

All generator settings live in the YAML. The synthetic labels are independent
of the transaction histories, so its AUC is **not a model-quality result**.

## Run with private data

For authorized local use, place the private parquet files in the locations specified by
[`configs/cashflow_moe.yaml`](configs/cashflow_moe.yaml), or copy that configuration
and update its `data` paths. The required columns and transaction sign convention
are documented in [the data contract](docs/data.md).

```bash
uv run cashflow-run --config configs/cashflow_moe.yaml
uv run cashflow-run --config configs/cashflow_moe_importance.yaml
```

The first configuration preserves the original 56-feature model and training
policy. The second uses the same training settings and measures all 56 features
with ten permutations each on the validation split. Historical `v15_4*.yaml`
configurations remain available for reproducibility.

Each run writes an isolated directory:

```text
logs/YYYYMMDD_HHMMSS_<experiment_name>/
  config.yaml                          effective experiment configuration
  source_config.yaml                   original supplied YAML file
  run.log                              execution log
  metadata.json                        code, environment, and input metadata
  metrics.json                         validation and test metrics
  feature_names.json                   ordered model input names
  splits.csv                           consumer-to-split assignments
  models/model.joblib                   fitted router and experts
  predictions/validation_predictions.csv
  predictions/test_predictions.csv
  permutation_importance.json          when enabled
```

Data, predictions, and fitted models are excluded from git. Real-data metrics and
run artifacts remain local; the public demo verifies execution and reproducibility.

## Reproduce the diagram

The architecture diagram is generated from source without loading consumer data:

```bash
uv sync --group figures
uv run --group figures python scripts/figures/generate.py --config configs/figures/paper.yaml
```

This exports a PNG preview, an editable SVG, and a one-page vector PDF.
See the [figure guide](docs/figures/README.md) for the diagram's scope and export settings.

## Code map

| Location | Responsibility |
|---|---|
| `configs/` | Experiment settings, demo generation, and figure export parameters |
| `scripts/figures/` | Reproducible architecture diagram |
| `src/cashflow_ml/data.py` | Parquet loading, input validation, and splitting |
| `src/cashflow_ml/features/` | Stateless feature families and named feature sets |
| `src/cashflow_ml/models/` | Router/expert model definition and prediction |
| `src/cashflow_ml/sampling.py` | Negative downsampling and positive augmentation |
| `src/cashflow_ml/trainer.py` | Training the router and experts |
| `src/cashflow_ml/evaluations/` | Metrics and permutation importance |
| `src/cashflow_ml/experiment.py` | Compose a run and persist its artifacts |
| `tests/` | Behavioral regression, data boundaries, and pipeline checks |

```bash
uv run pytest
uv run ruff check .
```

[`uv.lock`](uv.lock) pins the environment. See the
[refactor verification](docs/verification.md) for numerical equivalence checks, the
[engineering guide](docs/ML_PROJECT_GUIDE.md) for extension points, the
[methodology](docs/methodology.md) for modeling decisions, and the existing
[feature reference](docs/feature_explanation.md) for feature definitions and
transaction-category mappings.
