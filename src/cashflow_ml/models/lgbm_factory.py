"""Construct the shared zero-imputation + LightGBM architecture."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise ImportError(
        "cashflow-ml requires LightGBM. Run `uv sync` from the project root "
        "to install the locked environment."
    ) from exc


def build_lgbm(hp: dict, seed: int, n_jobs: int) -> Pipeline:
    """Create an unfitted pipeline; never silently substitute another model."""
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0).set_output(transform="pandas"),
            ),
            ("clf", LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **hp)),
        ]
    )
