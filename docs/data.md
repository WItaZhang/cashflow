# Data contract

All paths come from an experiment YAML and are resolved from the working
directory. Run commands from the repository root. Original parquet files remain
external to git and are read-only inputs to the experiment.

## Consumer table

`data.consumer_file` points to one parquet file with one row per consumer.

| Column | Meaning |
|---|---|
| `masked_consumer_id` | Unique string ID; the first three characters identify C01, C02, or C03 |
| `evaluation_date` | Parseable date or timestamp; the feature observation cutoff |
| `total_balance` | Numeric balance at evaluation time; negative values are allowed |
| `FPF_TARGET` | Binary label, 0 or 1; null labels are excluded from supervised splitting |

## Transaction shards

`data.transactions_dir` contains one or more `transactions_*.parquet` files.
Shards are read in filename order and concatenated.

| Column | Meaning |
|---|---|
| `masked_consumer_id` | Consumer ID joining to the consumer table |
| `masked_transaction_id` | Transaction identifier used in count aggregates |
| `posted_date` | Parseable transaction date or timestamp |
| `amount` | Signed numeric amount: positive inflow, negative outflow |
| `category` | Numeric category code from the source dataset |

The selected features use category codes 3 (paycheck), 10 (small-dollar advance),
12 (loan), 16 (general merchandise), 23 (account fees), and 26 (credit-card
payment), as well as aggregates across categories. The full category mapping is
in [the feature reference](feature_explanation.md).

Only transactions joining to a known consumer are used. A transaction is
eligible when `posted_date <= evaluation_date` for that consumer. No future
transaction is used to compute the selected features. This cutoff alone does
not establish that the target label was defined without leakage; label timing
must be verified against the original dataset documentation.

## Validation and missing data

The loader checks required columns and values before feature construction, so
malformed input fails near the data boundary. Missing files, empty shard
directories, duplicate consumer rows, invalid dates, and invalid labels should
be fixed at their source. Insufficient history can yield missing feature values;
model pipelines use constant-zero imputation for these values. Category pivots
preserve historical semantics: absent consumer/category pairs within a populated
pivot are zero, while a category missing globally is not synthesized. A selected
column missing globally therefore fails feature selection with an explicit error.
A source-data validation error is not silently converted into an imputed feature.

Each client needs enough labeled consumers for a stratified 60/20/20 split.
For meaningful group AUC, both classes must appear in the evaluated split.
Small groups can also fail the minimum-row or two-class requirement for fitting
an expert. The synthetic demo supplies enough examples of both labels in all
three groups.

## Storage boundaries

| Directory | Use |
|---|---|
| `data/raw/` | Original source data; never written by the pipeline |
| `data/processed/` | Reserved for reusable processed datasets |
| `data/staging/` | Synthetic demo inputs and temporary local data |
| `logs/` | Configuration snapshots, provenance, metrics, fitted models, and predictions |

The demo writes only inside its configured `demo.staging_root`. It reuses an
existing fixture only if its contents equal the freshly generated fixture, and
refuses conflicting files or extra transaction shards. After changing generator
settings, choose new `data` paths beneath staging. The demo never needs access
to the original dataset.
