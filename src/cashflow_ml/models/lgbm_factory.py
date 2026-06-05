from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def build_lgbm(hp: dict, seed: int, n_jobs: int) -> Pipeline:
    """LightGBM classifier with a stable sklearn fallback."""
    if HAS_LGBM:
        clf = LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **hp)
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=hp.get("n_estimators", 300),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=6,
            min_samples_leaf=hp.get("min_child_samples", 50),
            random_state=seed,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0).set_output(transform="pandas")),
        ("clf", clf),
    ])
