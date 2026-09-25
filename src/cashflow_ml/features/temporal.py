"""Monthly cashflow trends and amplitude-based Fourier indicators.

Monthly grids contain observed transaction months only, including months with
zero inflow. Inactive calendar months are not zero-filled: changing that grid
would change both slopes and FFT inputs. FFT ratios use amplitudes, not squared
power, to preserve the original trained feature definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cashflow_ml.features.common import EPS, safe_div


def _ts_slope_r2(monthly: pd.DataFrame, val_col: str, prefix: str) -> pd.DataFrame:
    m = monthly.copy()
    m["_x"] = m.groupby("masked_consumer_id").cumcount().astype(float)
    g = m.groupby("masked_consumer_id")
    mean_x = g["_x"].mean()
    mean_y = g[val_col].mean()
    m = m.join(mean_x.rename("_mx"), on="masked_consumer_id")
    m = m.join(mean_y.rename("_my"), on="masked_consumer_id")
    m["_dxdy"] = (m["_x"] - m["_mx"]) * (m[val_col] - m["_my"])
    m["_dx2"] = (m["_x"] - m["_mx"]) ** 2
    m["_dy2"] = (m[val_col] - m["_my"]) ** 2
    agg = m.groupby("masked_consumer_id").agg(
        _sxy=("_dxdy", "sum"),
        _sxx=("_dx2", "sum"),
        _syy=("_dy2", "sum"),
        _n=(val_col, "count"),
    )
    slope = safe_div(agg["_sxy"], agg["_sxx"])
    r2 = safe_div(agg["_sxy"] ** 2, agg["_sxx"] * agg["_syy"] + EPS)
    out = pd.DataFrame(
        {
            f"{prefix}_slope": slope,
            f"{prefix}_slope_norm": safe_div(slope, mean_y),
            f"{prefix}_r2": r2,
        }
    )
    out.loc[agg["_n"] < 2, list(out.columns)] = np.nan
    return out


def _ts_d2_mean(monthly: pd.DataFrame, val_col: str, prefix: str) -> pd.Series:
    m = monthly.copy()
    m["_d1"] = m.groupby("masked_consumer_id")[val_col].diff()
    m["_d2"] = m.groupby("masked_consumer_id")["_d1"].diff()
    return m.groupby("masked_consumer_id")["_d2"].mean().rename(f"{prefix}_d2_mean")


def build_monthly_series(
    tx: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological inflow, outflow, and net series per consumer."""
    tx_m = tx.copy()
    tx_m["_month"] = tx_m["posted_date"].values.astype("datetime64[M]")
    inflow_monthly = (
        tx_m.groupby(["masked_consumer_id", "_month"])["inflow_amount"]
        .sum()
        .reset_index()
        .sort_values(["masked_consumer_id", "_month"])
    )
    outflow_monthly = (
        tx_m.groupby(["masked_consumer_id", "_month"])["amount"]
        .apply(lambda amounts: (-amounts.clip(upper=0)).sum())
        .reset_index(name="outflow_amount")
        .sort_values(["masked_consumer_id", "_month"])
    )
    net_monthly = (
        tx_m.groupby(["masked_consumer_id", "_month"])["amount"]
        .sum()
        .reset_index()
        .sort_values(["masked_consumer_id", "_month"])
    )

    return inflow_monthly, outflow_monthly, net_monthly


def compute_trend_features(
    inflow_monthly: pd.DataFrame,
    outflow_monthly: pd.DataFrame,
    net_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Fit monthly slopes and average second differences without imputation."""
    ts_inflow_sr = _ts_slope_r2(inflow_monthly, "inflow_amount", "ts_inflow")
    ts_outflow_sr = _ts_slope_r2(outflow_monthly, "outflow_amount", "ts_outflow")
    ts_net_sr = _ts_slope_r2(net_monthly, "amount", "ts_net")
    ts_inflow_d2 = _ts_d2_mean(inflow_monthly, "inflow_amount", "ts_inflow")
    ts_outflow_d2 = _ts_d2_mean(outflow_monthly, "outflow_amount", "ts_outflow")
    ts_net_d2 = _ts_d2_mean(net_monthly, "amount", "ts_net")

    ts_all = ts_inflow_sr.join(ts_outflow_sr, how="outer").join(ts_net_sr, how="outer")
    for s in [ts_inflow_d2, ts_outflow_d2, ts_net_d2]:
        ts_all = ts_all.join(s, how="outer")
    ts_all = ts_all.drop(columns=["ts_outflow_r2", "ts_net_r2"], errors="ignore")
    return ts_all


def _fft_features_from_series(
    monthly: pd.DataFrame, val_col: str, prefix: str, min_months: int = 3
) -> pd.DataFrame:
    """
    Compute per-consumer FFT features from a monthly time series.

    Features (per series):
      dom_power_ratio    — dominant AC amplitude / total AC amplitude (periodicity strength)
      spectral_entropy   — entropy of normalized amplitude spectrum (disorder)
      dom_freq_idx       — which harmonic dominates (1 = 1 cycle total, 2 = 2 cycles, ...)
      low_freq_ratio     — amplitude at freq 1-2 / total AC amplitude (slow trend dominance)
      top2_power_ratio   — top-2 amplitudes / total AC amplitude
    """
    records = []
    cid_col = "masked_consumer_id"
    for cid, grp in monthly.groupby(cid_col):
        vals = grp[val_col].values.astype(float)
        n = len(vals)
        if n < min_months:
            records.append(
                {
                    cid_col: cid,
                    f"{prefix}_dom_power_ratio": np.nan,
                    f"{prefix}_spectral_entropy": np.nan,
                    f"{prefix}_dom_freq_idx": np.nan,
                    f"{prefix}_low_freq_ratio": np.nan,
                    f"{prefix}_top2_power_ratio": np.nan,
                }
            )
            continue

        # Zero-mean the series so DC doesn't swamp everything
        vals_zm = vals - vals.mean()
        fft_coeffs = np.fft.rfft(vals_zm)
        amplitudes = np.abs(fft_coeffs)
        # Drop DC (index 0) — we already subtracted the mean
        ac_amps = amplitudes[1:]
        total_ac = ac_amps.sum() + EPS

        dom_idx = int(np.argmax(ac_amps)) + 1  # 1-indexed into original spectrum
        dom_power_ratio = ac_amps[dom_idx - 1] / total_ac

        # Spectral entropy (using amplitudes as probabilities)
        p = ac_amps / total_ac
        spectral_entropy = float(-np.sum(p * np.log(p + EPS)))

        # Low-frequency ratio: freq indices 1 and 2 (slowest oscillations)
        low_freq = ac_amps[: min(2, len(ac_amps))].sum()
        low_freq_ratio = low_freq / total_ac

        # Top-2 amplitudes
        sorted_ac = np.sort(ac_amps)[::-1]
        top2 = sorted_ac[:2].sum() if len(sorted_ac) >= 2 else sorted_ac[0]
        top2_power_ratio = top2 / total_ac

        records.append(
            {
                cid_col: cid,
                f"{prefix}_dom_power_ratio": float(dom_power_ratio),
                f"{prefix}_spectral_entropy": float(spectral_entropy),
                f"{prefix}_dom_freq_idx": float(dom_idx),
                f"{prefix}_low_freq_ratio": float(low_freq_ratio),
                f"{prefix}_top2_power_ratio": float(top2_power_ratio),
            }
        )
    return pd.DataFrame(records).set_index(cid_col)


def compute_fft_features(
    inflow_monthly: pd.DataFrame,
    outflow_monthly: pd.DataFrame,
    net_monthly: pd.DataFrame,
) -> pd.DataFrame:
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
    fft_in = _fft_features_from_series(inflow_monthly, "inflow_amount", "fft_inflow")
    fft_out = _fft_features_from_series(outflow_monthly, "outflow_amount", "fft_outflow")
    fft_net = _fft_features_from_series(net_monthly, "amount", "fft_net")
    return fft_in.join(fft_out, how="outer").join(fft_net, how="outer")
