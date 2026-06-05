from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from cashflow_ml.models.lgbm_factory import build_lgbm


@dataclass
class SoftRoutingLgbmMoe:
    """LightGBM router + per-client LightGBM experts."""

    config: dict
    seed: int
    n_jobs: int
    router: object | None = None
    router_classes: np.ndarray | None = None
    experts: dict[str, object] = field(default_factory=dict)
    expert_cols: dict[str, list[str]] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "SoftRoutingLgbmMoe":
        self.feature_cols = list(feature_cols)
        X_train = train_df[self.feature_cols]
        y_train = train_df["FPF_TARGET"].astype(int)

        hp = self.config["hyperparameters"]
        self.router = build_lgbm(hp["router"], self.seed, self.n_jobs)
        router_y = train_df["masked_consumer_id"].str[:3].values
        self.router.fit(X_train, router_y)
        self.router_classes = np.array(self.router.named_steps["clf"].classes_)

        training = self.config.get("training", {})
        cid_train = train_df["masked_consumer_id"].str[:3].values
        rng = np.random.default_rng(self.seed)
        for cid, expert_hp in hp["experts"].items():
            mask = cid_train == cid
            sub_y = y_train.iloc[mask]
            if sub_y.nunique() < 2 or mask.sum() < training.get("min_expert_rows", 20):
                continue
            X_sub = X_train.iloc[mask].copy()
            y_sub = sub_y.copy()

            if cid in set(training.get("downsample_negative_clients", [])):
                X_sub, y_sub = self._downsample_negatives(
                    X_sub,
                    y_sub,
                    keep_ratio=training.get("downsample_negative_keep_ratio", 0.2),
                    rng=rng,
                )

            if cid in set(training.get("smote_positive_clients", [])):
                X_sub, y_sub = self._smote_positives(
                    X_sub,
                    y_sub,
                    positive_multiplier=training.get("smote_positive_multiplier", 2.0),
                    k_neighbors=training.get("smote_k_neighbors", 5),
                    rng=rng,
                )

            sample_weight = None
            if training.get("sample_weight", False):
                neg_w = 1.0 / training.get("downsample_negative_keep_ratio", 1.0)
                pos_w = 1.0 / training.get("smote_positive_multiplier", 1.0)
                sample_weight = np.where(y_sub.values == 0, neg_w, pos_w).astype(float)

            expert = build_lgbm(expert_hp, self.seed, self.n_jobs)
            if sample_weight is not None:
                expert.fit(X_sub, y_sub, clf__sample_weight=sample_weight)
            else:
                expert.fit(X_sub, y_sub)
            self.experts[cid] = expert
            self.expert_cols[cid] = self.feature_cols
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.router is None or self.router_classes is None:
            raise RuntimeError("Model is not fitted.")
        X = df[self.feature_cols]
        router_probs = self.router.predict_proba(X)
        out = np.zeros(len(X))
        total_w = np.zeros(len(X))
        for j, cid in enumerate(self.router_classes):
            if cid not in self.experts:
                continue
            cols = self.expert_cols[cid]
            p_expert = self.experts[cid].predict_proba(X[cols])[:, 1]
            out += router_probs[:, j] * p_expert
            total_w += router_probs[:, j]
        return out / np.maximum(total_w, 1e-9)

    @staticmethod
    def _downsample_negatives(
        X: pd.DataFrame,
        y: pd.Series,
        keep_ratio: float,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, pd.Series]:
        y_arr = y.values
        neg_idx = np.where(y_arr == 0)[0]
        pos_idx = np.where(y_arr == 1)[0]
        n_keep = int(round(len(neg_idx) * keep_ratio))
        keep_neg = rng.choice(neg_idx, size=n_keep, replace=False)
        selected = np.sort(np.concatenate([keep_neg, pos_idx]))
        return X.iloc[selected], y.iloc[selected]

    @staticmethod
    def _smote_positives(
        X: pd.DataFrame,
        y: pd.Series,
        positive_multiplier: float,
        k_neighbors: int,
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, pd.Series]:
        n_pos = int((y.values == 1).sum())
        n_synth = max(0, int(round(n_pos * positive_multiplier)) - n_pos)
        if n_synth <= 0 or n_pos < 2:
            return X, y

        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        X_imp = imputer.fit_transform(X)
        X_pos = X_imp[y.values == 1]
        k_eff = min(k_neighbors, n_pos - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_pos)
        _, nbr_idx = nn.kneighbors(X_pos)
        base = rng.integers(0, n_pos, size=n_synth)
        nbr_pick = rng.integers(1, k_eff + 1, size=n_synth)
        chosen = nbr_idx[base, nbr_pick]
        alpha = rng.random(size=(n_synth, 1))
        X_synth = X_pos[base] + alpha * (X_pos[chosen] - X_pos[base])

        original = pd.DataFrame(X_imp, columns=X.columns)
        synth = pd.DataFrame(X_synth, columns=X.columns)
        X_out = pd.concat([original, synth], ignore_index=True)
        y_out = pd.concat(
            [y.reset_index(drop=True), pd.Series([1] * n_synth, dtype=int)],
            ignore_index=True,
        )
        return X_out, y_out
