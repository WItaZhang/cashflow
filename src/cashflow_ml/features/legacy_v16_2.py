"""
evaluation_v16.py — v15 baseline + two new time-series feature families.

New features added on top of the 44 v15 features:

  Track 1 — FFT / spectral features (15 features, prefix fft_*)
    Fourier transform of monthly inflow / outflow / net series.
    Captures periodicity strength, dominant frequency, spectral entropy.

  Track 2 — Credit-risk behavioral TS features (16 features, prefix risk_*)
    Velocity / deceleration, spend concentration, income drought,
    round-amount ratio, overdraft months, last-3-month trajectory, etc.
    Inspired by published FPD / early-warning research on transactional data.

Usage:
    uv run python evaluation_v16.py [--n-folds 3] [--mode all|fft|risk]
"""
from __future__ import annotations

import os
from pathlib import Path

_N_CORES = "8"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = _N_CORES

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from cashflow_ml.evaluations.metrics import group_auc as compute_score

# ---------------------------------------------------------------------------
# Constants (same as v15)
# ---------------------------------------------------------------------------
HP_BASELINE    = {"n_estimators": 600, "learning_rate": 0.03,
                  "num_leaves": 63,  "min_child_samples": 30}
HP_REGULARIZED = {"n_estimators": 800, "learning_rate": 0.02,
                  "num_leaves": 31,  "min_child_samples": 60,
                  "reg_lambda": 1.0, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5}
HP_BALANCED    = {"n_estimators": 700, "learning_rate": 0.025,
                  "num_leaves": 47,  "min_child_samples": 45,
                  "reg_lambda": 0.5, "feature_fraction": 0.9,
                  "bagging_fraction": 0.9, "bagging_freq": 5}
MOE_STRATEGY   = {"C01": HP_BASELINE, "C02": HP_REGULARIZED, "C03": HP_BALANCED}
MIN_EXPERT     = 20

# v16 sample-weight compensation (same as evaluation_v16.py)
_DOWNSAMPLE_CLIENTS   = {"C01", "C03"}
_DOWNSAMPLE_KEEP      = 0.2          # keep 20 % of negatives
_NEG_W                = 1.0 / _DOWNSAMPLE_KEEP   # = 5.0
_SMOTE_CLIENTS        = {"C01", "C03"}
_SMOTE_POS_MULT       = 2.0
_SMOTE_K              = 5
_POS_W                = 1.0 / _SMOTE_POS_MULT    # = 0.5


def _smote_oversample(X_pos: np.ndarray, n_synth: int,
                      rng: np.random.Generator) -> np.ndarray:
    n_pos, n_feat = X_pos.shape
    if n_synth <= 0 or n_pos < 2:
        return np.empty((0, n_feat))
    k_eff = min(_SMOTE_K, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_pos)
    _, nbr_idx = nn.kneighbors(X_pos)
    base      = rng.integers(0, n_pos, size=n_synth)
    nbr_pick  = rng.integers(1, k_eff + 1, size=n_synth)
    chosen    = nbr_idx[base, nbr_pick]
    alpha     = rng.random(size=(n_synth, 1))
    return X_pos[base] + alpha * (X_pos[chosen] - X_pos[base])

# v15 baseline 44-feature list
V15_FEATURES: list[str] = [
    "total_balance",
    "amount_mean", "amount_std", "amount_abs_mean", "amount_abs_max",
    "inflow_sum", "inflow_sum_7d", "inflow_sum_30d",
    "txn_count_7d",
    "inflow_to_balance_ratio", "outflow_to_balance_ratio",
    "outflow_to_inflow_ratio",
    "recent_7d_to_30d_txn_ratio", "recent_30d_to_90d_txn_ratio",
    "all_cat_3_amount", "all_cat_12_amount", "all_cat_26_amount",
    "all_cat_26_count", "all_cat_3_count",
    "d30_cat_3_amount", "d30_cat_12_amount", "d30_cat_23_amount", "d30_cat_26_amount",
    "d90_cat_3_count", "d90_cat_10_count",
    "d90_cat_12_amount", "d90_cat_23_amount", "d90_cat_26_amount",
    "cc_payment_amount_share",
    "gen_merch_avg_txn_amount",
    "inflow_gap_mean", "inflow_gap_cv",
    "inflow_monthly_min",
    "history_days", "recency_days",
    "ts_inflow_slope", "ts_inflow_r2",
    "ts_outflow_slope", "ts_outflow_slope_norm",
    "ts_net_slope", "ts_net_slope_norm",
    "ts_inflow_d2_mean",
    "ts_outflow_d2_mean",
    "ts_net_d2_mean",
]

EPS = 1e-6


def _safe_div(a, b, eps=EPS):
    return a / (b + eps)


# ---------------------------------------------------------------------------
# Base feature engineering (v15 process_inputs, self-contained)
# ---------------------------------------------------------------------------
_CAT_ALL = {3, 12, 26}
_CAT_D30 = {3, 12, 23, 26}
_CAT_D90 = {3, 10, 12, 23, 26}


def _cat_pivot(tx, cats, prefix, want_count, want_amount):
    sub = tx[tx["category"].isin(cats)].copy()
    if sub.empty:
        return pd.DataFrame(index=pd.Index([], name="masked_consumer_id"))
    parts = []
    if want_count:
        cnt = (sub.pivot_table(index="masked_consumer_id", columns="category",
                               values="masked_transaction_id", aggfunc="count",
                               fill_value=0)
               .rename(columns=lambda c: f"{prefix}_cat_{int(c)}_count"))
        parts.append(cnt)
    if want_amount:
        amt = (sub.pivot_table(index="masked_consumer_id", columns="category",
                               values="amount_abs", aggfunc="sum", fill_value=0.0)
               .rename(columns=lambda c: f"{prefix}_cat_{int(c)}_amount"))
        parts.append(amt)
    result = parts[0]
    for p in parts[1:]:
        result = result.join(p, how="outer")
    return result


def _ts_slope_r2(monthly, val_col, prefix):
    m = monthly.copy()
    m["_x"] = m.groupby("masked_consumer_id").cumcount().astype(float)
    g = m.groupby("masked_consumer_id")
    mean_x = g["_x"].mean()
    mean_y = g[val_col].mean()
    m = m.join(mean_x.rename("_mx"), on="masked_consumer_id")
    m = m.join(mean_y.rename("_my"), on="masked_consumer_id")
    m["_dxdy"] = (m["_x"] - m["_mx"]) * (m[val_col] - m["_my"])
    m["_dx2"]  = (m["_x"] - m["_mx"]) ** 2
    m["_dy2"]  = (m[val_col] - m["_my"]) ** 2
    agg = m.groupby("masked_consumer_id").agg(
        _sxy=("_dxdy", "sum"), _sxx=("_dx2", "sum"), _syy=("_dy2", "sum"),
        _n=(val_col, "count"),
    )
    slope = _safe_div(agg["_sxy"], agg["_sxx"])
    r2    = _safe_div(agg["_sxy"] ** 2, agg["_sxx"] * agg["_syy"] + EPS)
    out   = pd.DataFrame({
        f"{prefix}_slope":      slope,
        f"{prefix}_slope_norm": _safe_div(slope, mean_y),
        f"{prefix}_r2":         r2,
    })
    out.loc[agg["_n"] < 2, list(out.columns)] = np.nan
    return out


def _ts_d2_mean(monthly, val_col, prefix):
    m = monthly.copy()
    m["_d1"] = m.groupby("masked_consumer_id")[val_col].diff()
    m["_d2"] = m.groupby("masked_consumer_id")["_d1"].diff()
    return m.groupby("masked_consumer_id")["_d2"].mean().rename(f"{prefix}_d2_mean")


def _base_process_inputs(consumer_file: str, transactions_dir: str) -> tuple[pd.DataFrame, dict]:
    """Reproduce v15 44-feature set, plus keep monthly series for new families.

    Returns:
        (features DataFrame, intermediates dict with keys:
         inflow_monthly, outflow_monthly, net_monthly, transactions, consumers)
    """
    consumers = pd.read_parquet(consumer_file).copy()
    consumers["evaluation_date"] = pd.to_datetime(consumers["evaluation_date"])

    txn_files = sorted(Path(transactions_dir).glob("transactions_*.parquet"))
    transactions = pd.concat([pd.read_parquet(f) for f in txn_files], ignore_index=True)
    transactions["posted_date"] = pd.to_datetime(transactions["posted_date"])

    tx = transactions.merge(
        consumers[["masked_consumer_id", "evaluation_date"]],
        on="masked_consumer_id", how="inner",
    )
    tx = tx[tx["posted_date"] <= tx["evaluation_date"]].copy()
    tx["amount_abs"]    = tx["amount"].abs()
    tx["inflow_amount"] = tx["amount"].clip(lower=0)
    tx["days_before"]   = (tx["evaluation_date"] - tx["posted_date"]).dt.days

    agg_all = tx.groupby("masked_consumer_id").agg(
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        amount_abs_mean=("amount_abs", "mean"),
        amount_abs_max=("amount_abs", "max"),
        inflow_sum=("inflow_amount", "sum"),
    )
    tx7  = tx[tx["days_before"] <= 7]
    tx30 = tx[tx["days_before"] <= 30]
    tx90 = tx[tx["days_before"] <= 90]
    agg_7  = tx7.groupby("masked_consumer_id").agg(
        inflow_sum_7d=("inflow_amount", "sum"),
        txn_count_7d=("masked_transaction_id", "count"),
    )
    agg_30 = tx30.groupby("masked_consumer_id").agg(
        inflow_sum_30d=("inflow_amount", "sum"),
        txn_count_30d=("masked_transaction_id", "count"),
    )
    agg_90 = tx90.groupby("masked_consumer_id").agg(
        txn_count_90d=("masked_transaction_id", "count"),
    )

    cat_all = _cat_pivot(tx,   _CAT_ALL, "all", want_count=True,  want_amount=True)
    cat_d30 = _cat_pivot(tx30, _CAT_D30, "d30", want_count=False, want_amount=True)
    cat_d90 = _cat_pivot(tx90, _CAT_D90, "d90", want_count=True,  want_amount=True)

    features = consumers[["masked_consumer_id", "total_balance", "FPF_TARGET",
                           "evaluation_date"]].copy()
    for frame in [agg_all, agg_7, agg_30, agg_90]:
        features = features.merge(frame.reset_index(), on="masked_consumer_id", how="left")
    for pivot in [cat_all, cat_d30, cat_d90]:
        if not pivot.empty:
            features = features.merge(pivot.reset_index(), on="masked_consumer_id", how="left")

    outflow_sum = (tx.groupby("masked_consumer_id")["amount"]
                     .apply(lambda s: (-s.clip(upper=0)).sum()))
    features = features.merge(outflow_sum.rename("_outflow_sum").reset_index(),
                              on="masked_consumer_id", how="left")
    features["_outflow_sum"] = features["_outflow_sum"].fillna(0.0)

    bal_abs = features["total_balance"].abs() + 1.0
    features["inflow_to_balance_ratio"]      = _safe_div(features["inflow_sum"], bal_abs)
    features["outflow_to_balance_ratio"]     = _safe_div(features["_outflow_sum"], bal_abs)
    features["outflow_to_inflow_ratio"]      = _safe_div(features["_outflow_sum"],
                                                         features["inflow_sum"])
    features["recent_7d_to_30d_txn_ratio"]  = _safe_div(features["txn_count_7d"],
                                                         features["txn_count_30d"])
    features["recent_30d_to_90d_txn_ratio"] = _safe_div(features["txn_count_30d"],
                                                         features["txn_count_90d"])

    cc_out = (tx[tx["category"] == 26]
              .groupby("masked_consumer_id")["amount_abs"].sum().rename("_cc"))
    features = features.merge(cc_out.reset_index(), on="masked_consumer_id", how="left")
    features["_cc"] = features["_cc"].fillna(0.0)
    features["cc_payment_amount_share"] = _safe_div(features["_cc"], features["_outflow_sum"])

    gm = (tx[tx["category"] == 16].groupby("masked_consumer_id")
          .agg(_gm_count=("masked_transaction_id", "count"),
               _gm_amount=("amount_abs", "sum")))
    features = features.merge(gm.reset_index(), on="masked_consumer_id", how="left")
    features[["_gm_count", "_gm_amount"]] = features[["_gm_count", "_gm_amount"]].fillna(0)
    features["gen_merch_avg_txn_amount"] = _safe_div(features["_gm_amount"],
                                                     features["_gm_count"])

    inflow_tx = tx[tx["inflow_amount"] > 0].copy()
    if not inflow_tx.empty:
        inflow_tx["_month"] = inflow_tx["posted_date"].values.astype("datetime64[M]")
        monthly_in = (inflow_tx.groupby(["masked_consumer_id", "_month"])
                                ["inflow_amount"].sum().reset_index()
                                .sort_values(["masked_consumer_id", "_month"]))
        monthly_min = monthly_in.groupby("masked_consumer_id")["inflow_amount"].min().rename(
            "inflow_monthly_min")
        inflow_srt = inflow_tx.sort_values(["masked_consumer_id", "posted_date"])
        inflow_srt["_gap"] = inflow_srt.groupby("masked_consumer_id")["posted_date"].diff().dt.days
        gaps = inflow_srt.groupby("masked_consumer_id")["_gap"].agg(
            inflow_gap_mean="mean", inflow_gap_std="std")
        gaps["inflow_gap_cv"] = _safe_div(gaps["inflow_gap_std"], gaps["inflow_gap_mean"])
        features = (features
                    .merge(monthly_min.reset_index(), on="masked_consumer_id", how="left")
                    .merge(gaps[["inflow_gap_mean", "inflow_gap_cv"]].reset_index(),
                           on="masked_consumer_id", how="left"))

    conf = tx.groupby("masked_consumer_id")["posted_date"].agg(_first="min", _last="max")
    conf = conf.merge(consumers[["masked_consumer_id", "evaluation_date"]],
                      on="masked_consumer_id", how="left")
    conf["history_days"] = (conf["evaluation_date"] - conf["_first"]).dt.days.clip(lower=0)
    conf["recency_days"] = (conf["evaluation_date"] - conf["_last"]).dt.days.clip(lower=0)
    features = features.merge(
        conf[["masked_consumer_id", "history_days", "recency_days"]],
        on="masked_consumer_id", how="left")

    tx_m = tx.copy()
    tx_m["_month"] = tx_m["posted_date"].values.astype("datetime64[M]")
    inflow_monthly  = (tx_m.groupby(["masked_consumer_id", "_month"])["inflow_amount"]
                            .sum().reset_index().sort_values(["masked_consumer_id", "_month"]))
    outflow_monthly = (tx_m.groupby(["masked_consumer_id", "_month"])
                            .apply(lambda g: (-g["amount"].clip(upper=0)).sum())
                            .reset_index(name="outflow_amount")
                            .sort_values(["masked_consumer_id", "_month"]))
    net_monthly     = (tx_m.groupby(["masked_consumer_id", "_month"])["amount"]
                            .sum().reset_index().sort_values(["masked_consumer_id", "_month"]))

    ts_inflow_sr  = _ts_slope_r2(inflow_monthly,  "inflow_amount",  "ts_inflow")
    ts_outflow_sr = _ts_slope_r2(outflow_monthly, "outflow_amount", "ts_outflow")
    ts_net_sr     = _ts_slope_r2(net_monthly,      "amount",         "ts_net")
    ts_inflow_d2  = _ts_d2_mean(inflow_monthly,   "inflow_amount",  "ts_inflow")
    ts_outflow_d2 = _ts_d2_mean(outflow_monthly,  "outflow_amount", "ts_outflow")
    ts_net_d2     = _ts_d2_mean(net_monthly,       "amount",         "ts_net")

    ts_all = ts_inflow_sr.join(ts_outflow_sr, how="outer").join(ts_net_sr, how="outer")
    for s in [ts_inflow_d2, ts_outflow_d2, ts_net_d2]:
        ts_all = ts_all.join(s, how="outer")
    ts_all = ts_all.drop(columns=["ts_outflow_r2", "ts_net_r2"], errors="ignore")
    features = features.merge(ts_all.reset_index(), on="masked_consumer_id", how="left")

    features.drop(columns=[c for c in features.columns if c.startswith("_")], inplace=True)

    intermediates = {
        "inflow_monthly": inflow_monthly,
        "outflow_monthly": outflow_monthly,
        "net_monthly": net_monthly,
        "transactions": tx,
        "consumers": consumers,
    }
    return features, intermediates


# ---------------------------------------------------------------------------
# Track 1: FFT / Spectral Features (15 features)
# ---------------------------------------------------------------------------

def _fft_features_from_series(monthly: pd.DataFrame, val_col: str, prefix: str,
                               min_months: int = 3) -> pd.DataFrame:
    """
    Compute per-consumer FFT features from a monthly time series.

    Features (per series):
      dom_power_ratio    — dominant AC amplitude / total AC amplitude (periodicity strength)
      spectral_entropy   — entropy of normalized amplitude spectrum (disorder)
      dom_freq_idx       — which harmonic dominates (1 = 1 cycle total, 2 = 2 cycles, ...)
      low_freq_ratio     — power at freq 1-2 / total AC power (slow trend dominance)
      top2_power_ratio   — top-2 amplitudes / total AC amplitude
    """
    records = []
    cid_col = "masked_consumer_id"
    for cid, grp in monthly.groupby(cid_col):
        vals = grp[val_col].values.astype(float)
        n = len(vals)
        if n < min_months:
            records.append({cid_col: cid,
                             f"{prefix}_dom_power_ratio": np.nan,
                             f"{prefix}_spectral_entropy": np.nan,
                             f"{prefix}_dom_freq_idx": np.nan,
                             f"{prefix}_low_freq_ratio": np.nan,
                             f"{prefix}_top2_power_ratio": np.nan})
            continue

        # Zero-mean the series so DC doesn't swamp everything
        vals_zm = vals - vals.mean()
        fft_coeffs = np.fft.rfft(vals_zm)
        amplitudes = np.abs(fft_coeffs)
        # Drop DC (index 0) — we already subtracted the mean
        ac_amps = amplitudes[1:]
        total_ac = ac_amps.sum() + EPS

        dom_idx = int(np.argmax(ac_amps)) + 1   # 1-indexed into original spectrum
        dom_power_ratio = ac_amps[dom_idx - 1] / total_ac

        # Spectral entropy (using amplitudes as probabilities)
        p = ac_amps / total_ac
        spectral_entropy = float(-np.sum(p * np.log(p + EPS)))

        # Low-frequency ratio: freq indices 1 and 2 (slowest oscillations)
        low_freq = ac_amps[:min(2, len(ac_amps))].sum()
        low_freq_ratio = low_freq / total_ac

        # Top-2 amplitudes
        sorted_ac = np.sort(ac_amps)[::-1]
        top2 = sorted_ac[:2].sum() if len(sorted_ac) >= 2 else sorted_ac[0]
        top2_power_ratio = top2 / total_ac

        records.append({
            cid_col:                          cid,
            f"{prefix}_dom_power_ratio":      float(dom_power_ratio),
            f"{prefix}_spectral_entropy":     float(spectral_entropy),
            f"{prefix}_dom_freq_idx":         float(dom_idx),
            f"{prefix}_low_freq_ratio":       float(low_freq_ratio),
            f"{prefix}_top2_power_ratio":     float(top2_power_ratio),
        })
    return pd.DataFrame(records).set_index(cid_col)


def compute_fft_features(inflow_monthly, outflow_monthly, net_monthly) -> pd.DataFrame:
    """
    15 FFT/spectral features:
      fft_inflow_dom_power_ratio     (1)
      fft_inflow_spectral_entropy    (2)
      fft_inflow_dom_freq_idx        (3)
      fft_inflow_low_freq_ratio      (4)
      fft_inflow_top2_power_ratio    (5)
      fft_outflow_dom_power_ratio    (6)
      fft_outflow_spectral_entropy   (7)
      fft_outflow_dom_freq_idx       (8)
      fft_outflow_low_freq_ratio     (9)
      fft_outflow_top2_power_ratio   (10)
      fft_net_dom_power_ratio        (11)
      fft_net_spectral_entropy       (12)
      fft_net_dom_freq_idx           (13)
      fft_net_low_freq_ratio         (14)
      fft_net_top2_power_ratio       (15)
    """
    fft_in  = _fft_features_from_series(inflow_monthly,  "inflow_amount",  "fft_inflow")
    fft_out = _fft_features_from_series(outflow_monthly, "outflow_amount", "fft_outflow")
    fft_net = _fft_features_from_series(net_monthly,     "amount",         "fft_net")
    return fft_in.join(fft_out, how="outer").join(fft_net, how="outer")


FFT_FEATURES: list[str] = [
    "fft_inflow_dom_power_ratio", "fft_inflow_spectral_entropy",
    "fft_inflow_dom_freq_idx", "fft_inflow_low_freq_ratio", "fft_inflow_top2_power_ratio",
    "fft_outflow_dom_power_ratio", "fft_outflow_spectral_entropy",
    "fft_outflow_dom_freq_idx", "fft_outflow_low_freq_ratio", "fft_outflow_top2_power_ratio",
    "fft_net_dom_power_ratio", "fft_net_spectral_entropy",
    "fft_net_dom_freq_idx", "fft_net_low_freq_ratio", "fft_net_top2_power_ratio",
]


# ---------------------------------------------------------------------------
# Track 2: Credit-Risk Behavioral TS Features (16 features)
# ---------------------------------------------------------------------------

def compute_risk_ts_features(tx: pd.DataFrame,
                              inflow_monthly: pd.DataFrame,
                              outflow_monthly: pd.DataFrame,
                              net_monthly: pd.DataFrame,
                              consumers: pd.DataFrame) -> pd.DataFrame:
    """
    16 credit-risk time-series features inspired by FPD / early-warning research:

      risk_velocity_inflow_recent     (1) — recent inflow rate vs. lifetime rate
      risk_velocity_outflow_recent    (2) — recent outflow rate vs. lifetime rate
      risk_velocity_txn_recent        (3) — recent txn rate vs. lifetime rate
      risk_net_negative_month_ratio   (4) — fraction of months spending > earning
      risk_inflow_monthly_skew        (5) — skewness of monthly inflow amounts
      risk_max_inflow_to_mean         (6) — peak monthly inflow / mean monthly inflow
      risk_last3m_inflow_vs_prior3m   (7) — last-3m inflow / prior-3m inflow (decline = risk)
      risk_last3m_outflow_vs_prior3m  (8) — last-3m outflow / prior-3m outflow
      risk_months_no_inflow_ratio     (9) — fraction of history months with zero inflow
      risk_round_amount_ratio         (10) — fraction of txns with round amounts (ATM/cash signal)
      risk_weekend_txn_ratio          (11) — fraction of transactions on weekends
      risk_category_hhi               (12) — Herfindahl index of spending by category
      risk_overdraft_months_ratio     (13) — fraction of months where outflow > inflow
      risk_inflow_gap_max_norm        (14) — max inflow gap (days) / history_days
      risk_late_month_txn_ratio       (15) — fraction of txns in last 5 days of each month
      risk_outflow_to_inflow_last3m   (16) — outflow/inflow ratio in last 3 months
    """
    cid_col = "masked_consumer_id"

    # --- velocity features: recent rate / lifetime rate ----------------------
    # recent = last 30 days; lifetime = history
    conf = tx.groupby(cid_col)["posted_date"].agg(_first="min")
    conf = conf.merge(consumers[[cid_col, "evaluation_date"]], on=cid_col, how="left")
    conf["history_days"] = (conf["evaluation_date"] - conf["_first"]).dt.days.clip(lower=1)

    tx30 = tx[tx["days_before"] <= 30]
    tx_recent_in  = tx30.groupby(cid_col)["inflow_amount"].sum()
    tx_recent_out = tx30.groupby(cid_col).apply(lambda g: (-g["amount"].clip(upper=0)).sum())
    tx_recent_cnt = tx30.groupby(cid_col)["masked_transaction_id"].count()
    tx_all_in  = tx.groupby(cid_col)["inflow_amount"].sum()
    tx_all_out = tx.groupby(cid_col).apply(lambda g: (-g["amount"].clip(upper=0)).sum())
    tx_all_cnt = tx.groupby(cid_col)["masked_transaction_id"].count()

    vel = pd.DataFrame(index=conf.set_index(cid_col).index)
    vel["_hist"] = conf.set_index(cid_col)["history_days"]
    vel["_in_rate_life"]   = _safe_div(tx_all_in,  vel["_hist"])
    vel["_out_rate_life"]  = _safe_div(tx_all_out, vel["_hist"])
    vel["_cnt_rate_life"]  = _safe_div(tx_all_cnt, vel["_hist"])
    vel["_in_rate_30d"]    = _safe_div(tx_recent_in,  30)
    vel["_out_rate_30d"]   = _safe_div(tx_recent_out, 30)
    vel["_cnt_rate_30d"]   = _safe_div(tx_recent_cnt, 30)
    vel["risk_velocity_inflow_recent"]  = _safe_div(vel["_in_rate_30d"],  vel["_in_rate_life"])
    vel["risk_velocity_outflow_recent"] = _safe_div(vel["_out_rate_30d"], vel["_out_rate_life"])
    vel["risk_velocity_txn_recent"]     = _safe_div(vel["_cnt_rate_30d"], vel["_cnt_rate_life"])

    # --- net negative month ratio --------------------------------------------
    both = (inflow_monthly[["masked_consumer_id", "_month", "inflow_amount"]]
            .merge(outflow_monthly[["masked_consumer_id", "_month", "outflow_amount"]],
                   on=["masked_consumer_id", "_month"], how="outer")
            .fillna(0.0))
    both["_net_pos"] = (both["outflow_amount"] > both["inflow_amount"]).astype(int)
    overdraft_ratio = both.groupby(cid_col).agg(
        risk_overdraft_months_ratio=("_net_pos", "mean"),
    )

    # --- inflow monthly stats -------------------------------------------------
    in_m_stats = inflow_monthly.groupby(cid_col)["inflow_amount"].agg(
        _mean="mean", _max="max",
    )
    in_m_stats["risk_max_inflow_to_mean"] = _safe_div(in_m_stats["_max"],
                                                       in_m_stats["_mean"])

    def _skew(vals):
        if len(vals) < 3:
            return np.nan
        return float(scipy_skew(vals))

    in_m_skew = inflow_monthly.groupby(cid_col)["inflow_amount"].apply(_skew).rename(
        "risk_inflow_monthly_skew")

    # --- fraction of history months with zero inflow -------------------------
    tx_ym = tx.copy()
    tx_ym["_month"] = tx_ym["posted_date"].values.astype("datetime64[M]")
    total_active_months = tx_ym.groupby(cid_col)["_month"].nunique().rename("_total_months")
    inflow_months_count = (inflow_monthly.groupby(cid_col)["inflow_amount"]
                           .apply(lambda s: (s > 0).sum()).rename("_inflow_months"))
    months_no_in = pd.DataFrame({"_total": total_active_months,
                                  "_inflow": inflow_months_count}).fillna(0)
    months_no_in["risk_months_no_inflow_ratio"] = _safe_div(
        months_no_in["_total"] - months_no_in["_inflow"], months_no_in["_total"])

    # --- last-3m vs prior-3m inflow/outflow ----------------------------------
    tx_m = tx.copy()
    tx_m["_month"] = tx_m["posted_date"].values.astype("datetime64[M]")
    eval_month = consumers[[cid_col, "evaluation_date"]].copy()
    eval_month["_eval_month"] = eval_month["evaluation_date"].values.astype("datetime64[M]")
    tx_m = tx_m.merge(eval_month[[cid_col, "_eval_month"]], on=cid_col, how="left")
    # months_ago = 0 means current month
    tx_m["_months_ago"] = ((tx_m["_eval_month"].astype("int64")
                            - tx_m["_month"].astype("int64"))
                           // (1_000_000_000 * 60 * 60 * 24 * 30))
    tx_last3  = tx_m[tx_m["_months_ago"] < 3]
    tx_prior3 = tx_m[(tx_m["_months_ago"] >= 3) & (tx_m["_months_ago"] < 6)]

    last3_in  = tx_last3.groupby(cid_col)["inflow_amount"].sum()
    prior3_in = tx_prior3.groupby(cid_col)["inflow_amount"].sum()
    last3_out  = tx_last3.groupby(cid_col).apply(lambda g: (-g["amount"].clip(upper=0)).sum())
    prior3_out = tx_prior3.groupby(cid_col).apply(lambda g: (-g["amount"].clip(upper=0)).sum())

    last3m_ratio_in  = _safe_div(last3_in,  prior3_in.reindex(last3_in.index).fillna(EPS))
    last3m_ratio_out = _safe_div(last3_out, prior3_out.reindex(last3_out.index).fillna(EPS))
    last3m_ratio_in.name  = "risk_last3m_inflow_vs_prior3m"
    last3m_ratio_out.name = "risk_last3m_outflow_vs_prior3m"

    last3m_out_to_in = _safe_div(last3_out, last3_in.reindex(last3_out.index).fillna(EPS))
    last3m_out_to_in.name = "risk_outflow_to_inflow_last3m"

    # --- round amount ratio --------------------------------------------------
    tx_copy = tx.copy()
    tx_copy["_is_round"] = ((tx_copy["amount_abs"] % 50 < EPS) |
                             (tx_copy["amount_abs"] % 100 < EPS)).astype(int)
    round_ratio = tx_copy.groupby(cid_col).agg(
        _round=("_is_round", "sum"), _total=("masked_transaction_id", "count")
    )
    round_ratio["risk_round_amount_ratio"] = _safe_div(round_ratio["_round"],
                                                        round_ratio["_total"])

    # --- weekend transaction ratio -------------------------------------------
    tx_copy["_is_weekend"] = tx_copy["posted_date"].dt.dayofweek.isin([5, 6]).astype(int)
    weekend_ratio = tx_copy.groupby(cid_col).agg(
        _wknd=("_is_weekend", "sum"), _tot=("masked_transaction_id", "count")
    )
    weekend_ratio["risk_weekend_txn_ratio"] = _safe_div(weekend_ratio["_wknd"],
                                                         weekend_ratio["_tot"])

    # --- category HHI --------------------------------------------------------
    cat_spend = tx_copy.groupby([cid_col, "category"])["amount_abs"].sum()
    cat_total = tx_copy.groupby(cid_col)["amount_abs"].sum()

    def _hhi(group):
        cid = group.index.get_level_values(cid_col)[0]
        shares = group.values / (cat_total.loc[cid] + EPS)
        return float((shares ** 2).sum())

    hhi = cat_spend.groupby(level=0).apply(_hhi).rename("risk_category_hhi")

    # --- max inflow gap normalized by history --------------------------------
    inflow_tx = tx[tx["inflow_amount"] > 0].copy()
    inflow_sorted = inflow_tx.sort_values([cid_col, "posted_date"])
    inflow_sorted["_gap"] = inflow_sorted.groupby(cid_col)["posted_date"].diff().dt.days
    max_gap = inflow_sorted.groupby(cid_col)["_gap"].max().rename("_max_gap")
    hist_d  = conf.set_index(cid_col)["history_days"]
    gap_norm = _safe_div(max_gap, hist_d.reindex(max_gap.index).clip(lower=1))
    gap_norm.name = "risk_inflow_gap_max_norm"

    # --- late-month transaction ratio ----------------------------------------
    tx_copy["_day_of_month"] = tx_copy["posted_date"].dt.day
    tx_copy["_is_late"] = (tx_copy["_day_of_month"] >= 25).astype(int)
    late_ratio = tx_copy.groupby(cid_col).agg(
        _late=("_is_late", "sum"), _tota=("masked_transaction_id", "count")
    )
    late_ratio["risk_late_month_txn_ratio"] = _safe_div(late_ratio["_late"],
                                                         late_ratio["_tota"])

    # --- assemble -----------------------------------------------------------
    out = (vel[["risk_velocity_inflow_recent",
                "risk_velocity_outflow_recent",
                "risk_velocity_txn_recent"]]
           .join(overdraft_ratio, how="outer")
           .join(in_m_stats[["risk_max_inflow_to_mean"]], how="outer")
           .join(in_m_skew, how="outer")
           .join(months_no_in[["risk_months_no_inflow_ratio"]], how="outer")
           .join(last3m_ratio_in, how="outer")
           .join(last3m_ratio_out, how="outer")
           .join(last3m_out_to_in, how="outer")
           .join(round_ratio[["risk_round_amount_ratio"]], how="outer")
           .join(weekend_ratio[["risk_weekend_txn_ratio"]], how="outer")
           .join(hhi, how="outer")
           .join(gap_norm, how="outer")
           .join(late_ratio[["risk_late_month_txn_ratio"]], how="outer"))
    out = out.drop(columns=[cid_col], errors="ignore")
    return out


RISK_TS_FEATURES: list[str] = [
    "risk_velocity_inflow_recent", "risk_velocity_outflow_recent",
    "risk_velocity_txn_recent",
    "risk_net_negative_month_ratio", "risk_inflow_monthly_skew",
    "risk_max_inflow_to_mean",
    "risk_last3m_inflow_vs_prior3m", "risk_last3m_outflow_vs_prior3m",
    "risk_months_no_inflow_ratio",
    "risk_round_amount_ratio", "risk_weekend_txn_ratio",
    "risk_category_hhi",
    "risk_overdraft_months_ratio",
    "risk_inflow_gap_max_norm",
    "risk_late_month_txn_ratio",
    "risk_outflow_to_inflow_last3m",
]


# ---------------------------------------------------------------------------
# Build model / MoE
# ---------------------------------------------------------------------------

def _build_lgbm(hp: dict) -> Pipeline:
    if _HAS_LGBM:
        clf = LGBMClassifier(random_state=42, n_jobs=8, verbose=-1, **hp)
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=hp.get("n_estimators", 300),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=6, min_samples_leaf=hp.get("min_child_samples", 50),
            random_state=42)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)
                    .set_output(transform="pandas")),
        ("clf", clf),
    ])


def _stratified_622(df: pd.DataFrame, seed: int):
    tr_p, va_p, te_p = [], [], []
    for _, sub in df.groupby(df["masked_consumer_id"].str[:3], sort=False):
        strat = sub["FPF_TARGET"] if sub["FPF_TARGET"].nunique() > 1 else None
        tv, te = train_test_split(sub, test_size=0.2, random_state=seed, stratify=strat)
        strat2 = tv["FPF_TARGET"] if tv["FPF_TARGET"].nunique() > 1 else None
        tr, va = train_test_split(tv, test_size=0.25, random_state=seed, stratify=strat2)
        tr_p.append(tr); va_p.append(va); te_p.append(te)
    return (pd.concat(tr_p).reset_index(drop=True),
            pd.concat(va_p).reset_index(drop=True),
            pd.concat(te_p).reset_index(drop=True))


def _moe_eval(labeled: pd.DataFrame, feature_cols: list[str], seed: int) -> dict:
    train_df, val_df, test_df = _stratified_622(labeled, seed)
    feats = [f for f in feature_cols if f in labeled.columns]

    X_tr = train_df[feats]; y_tr = train_df["FPF_TARGET"].astype(int)
    X_va = val_df[feats];   y_va = val_df["FPF_TARGET"].astype(int)
    X_te = test_df[feats];  y_te = test_df["FPF_TARGET"].astype(int)

    # Router
    router_y = train_df["masked_consumer_id"].str[:3].values
    router = _build_lgbm(HP_BASELINE)
    router.fit(X_tr, router_y)
    router_classes = np.array(router.named_steps["clf"].classes_)

    # Experts (v16: downsample + SMOTE + both sample_weight compensations)
    experts, expert_cols = {}, {}
    cid_tr = train_df["masked_consumer_id"].str[:3].values
    rng = np.random.default_rng(seed)
    for cid, hp in MOE_STRATEGY.items():
        mask = (cid_tr == cid)
        sub_y = y_tr.iloc[mask]
        if sub_y.nunique() < 2 or mask.sum() < MIN_EXPERT:
            continue
        X_sub = X_tr.iloc[mask].copy()

        # --- downsample negatives --------------------------------------------
        if cid in _DOWNSAMPLE_CLIENTS:
            y_arr    = sub_y.values
            neg_idx  = np.where(y_arr == 0)[0]
            pos_idx  = np.where(y_arr == 1)[0]
            n_keep   = int(round(len(neg_idx) * _DOWNSAMPLE_KEEP))
            keep_neg = rng.choice(neg_idx, size=n_keep, replace=False)
            sel      = np.sort(np.concatenate([keep_neg, pos_idx]))
            X_sub    = X_sub.iloc[sel]
            sub_y    = sub_y.iloc[sel]

        # --- SMOTE positives -------------------------------------------------
        if cid in _SMOTE_CLIENTS:
            n_pos_orig = int((sub_y.values == 1).sum())
            n_synth    = max(0, int(round(n_pos_orig * _SMOTE_POS_MULT)) - n_pos_orig)
            if n_synth > 0 and n_pos_orig >= 2:
                imputer  = SimpleImputer(strategy="constant", fill_value=0.0)
                X_imp    = imputer.fit_transform(X_sub)
                X_synth  = _smote_oversample(X_imp[sub_y.values == 1], n_synth, rng)
                synth_df = pd.DataFrame(X_synth, columns=X_sub.columns)
                X_sub    = pd.concat(
                    [pd.DataFrame(X_imp, columns=X_sub.columns), synth_df],
                    ignore_index=True)
                sub_y    = pd.concat(
                    [sub_y.reset_index(drop=True),
                     pd.Series([1] * n_synth, dtype=int)],
                    ignore_index=True)

        # --- sample weights: neg_w=5.0, pos_w=0.5 ---------------------------
        sw = None
        if cid in _DOWNSAMPLE_CLIENTS or cid in _SMOTE_CLIENTS:
            sw = np.where(sub_y.values == 0, _NEG_W, _POS_W).astype(float)

        expert = _build_lgbm(hp)
        if sw is None:
            expert.fit(X_sub, sub_y)
        else:
            expert.fit(X_sub, sub_y, clf__sample_weight=sw)
        experts[cid] = expert
        expert_cols[cid] = feats

    def _predict(X):
        probs = router.predict_proba(X)
        out = np.zeros(len(X)); total_w = np.zeros(len(X))
        for j, cid in enumerate(router_classes):
            if cid not in experts:
                continue
            p_e = experts[cid].predict_proba(X[expert_cols[cid]])[:, 1]
            out += probs[:, j] * p_e
            total_w += probs[:, j]
        return out / np.maximum(total_w, 1e-9)

    te_p = _predict(X_te)
    va_p = _predict(X_va)
    te_group = compute_score(test_df, te_p)
    va_group = compute_score(val_df,  va_p)

    cids_te = test_df["masked_consumer_id"].str[:3].values
    per_client = {}
    for cid in np.unique(cids_te):
        m = (cids_te == cid)
        if y_te.iloc[m].nunique() < 2:
            continue
        per_client[cid] = roc_auc_score(y_te.iloc[m], te_p[m])

    return {
        "test_group":   te_group,
        "val_group":    va_group,
        "test_auc":     roc_auc_score(y_te, te_p),
        "per_client":   per_client,
        "n_features":   len(feats),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _avg(fold_results: list[dict]) -> dict:
    scalars = ["test_group", "val_group", "test_auc"]
    avg = {k: float(np.mean([r[k] for r in fold_results])) for k in scalars}
    avg["std_test_group"] = float(np.std([r["test_group"] for r in fold_results]))
    avg["n_features"] = fold_results[0]["n_features"]
    all_cids = sorted({cid for r in fold_results for cid in r["per_client"]})
    avg["per_client"] = {}
    for cid in all_cids:
        vals = [r["per_client"][cid] for r in fold_results if cid in r["per_client"]]
        avg["per_client"][cid] = float(np.mean(vals)) if vals else np.nan
    return avg


def _print_avg(label: str, avg: dict) -> None:
    log.info(f"\n{'='*60}")
    log.info(f"  {label}")
    log.info(f"  n_features = {avg['n_features']}")
    log.info(f"  val   group = {avg['val_group']:.4f}")
    log.info(f"  test  group = {avg['test_group']:.4f}  (std={avg['std_test_group']:.4f})")
    log.info(f"  test  AUC   = {avg['test_auc']:.4f}")
    log.info(f"  per-client  = { {k: round(v,4) for k,v in avg['per_client'].items()} }")


def run_eval(n_folds: int = 3, mode: str = "all") -> None:
    log.info(f"=== evaluation_v16  mode={mode}  n_folds={n_folds}  [{_TS}] ===")
    log.info("Loading data and computing base v15 features ...")

    consumers = pd.read_parquet(CONSUMER_FILE).copy()
    consumers["evaluation_date"] = pd.to_datetime(consumers["evaluation_date"])

    base, _intermediates = _base_process_inputs(CONSUMER_FILE, TRANSACTIONS_DIR)

    inflow_monthly  = _intermediates["inflow_monthly"]
    outflow_monthly = _intermediates["outflow_monthly"]
    net_monthly     = _intermediates["net_monthly"]
    tx              = _intermediates["transactions"]

    log.info("Computing FFT features ...")
    fft_df = compute_fft_features(inflow_monthly, outflow_monthly, net_monthly)

    log.info("Computing credit-risk TS features ...")
    risk_df = compute_risk_ts_features(tx, inflow_monthly, outflow_monthly, net_monthly,
                                       consumers)

    # Merge new features into base frame
    base = base.merge(fft_df.reset_index(), on="masked_consumer_id", how="left")
    base = base.merge(risk_df.reset_index(), on="masked_consumer_id", how="left")

    labeled = base[base["FPF_TARGET"].notna()].copy()
    log.info(f"  labeled rows : {len(labeled)}")

    fold_seeds = [42, 123, 7][:n_folds]

    configs = {}
    if mode in ("all", "baseline"):
        configs["v15 baseline (44 features)"] = V15_FEATURES
    if mode in ("all", "fft"):
        configs["v15 + FFT (44+15=59 features)"] = V15_FEATURES + FFT_FEATURES
    if mode in ("all", "risk"):
        configs["v15 + risk_ts (44+16=60 features)"] = V15_FEATURES + RISK_TS_FEATURES
    if mode == "all":
        configs["v15 + FFT + risk_ts (44+31=75 features)"] = (
            V15_FEATURES + FFT_FEATURES + RISK_TS_FEATURES)

    for label, feat_list in configs.items():
        log.info(f"\n--- {label} ---")
        fold_results = []
        for i, seed in enumerate(fold_seeds):
            log.info(f"  fold {i+1}/{n_folds}  seed={seed} ...")
            r = _moe_eval(labeled.copy(), feat_list, seed)
            log.info(f"    test_group={r['test_group']:.4f}  n_feats={r['n_features']}")
            fold_results.append(r)
        avg = _avg(fold_results)
        _print_avg(label, avg)

    log.info(f"\nLog saved: {_LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--mode",    type=str, default="all",
                        choices=["all", "baseline", "fft", "risk"])
    args = parser.parse_args()
    run_eval(args.n_folds, args.mode)
