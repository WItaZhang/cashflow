"""Named feature sets used by experiment configs."""

V15_BASELINE_44 = [
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
    "ts_inflow_d2_mean", "ts_outflow_d2_mean", "ts_net_d2_mean",
]

V15_4_FFT_6 = [
    "fft_inflow_dom_power_ratio",
    "fft_outflow_dom_freq_idx",
    "fft_outflow_low_freq_ratio",
    "fft_inflow_low_freq_ratio",
    "fft_inflow_spectral_entropy",
    "fft_inflow_dom_freq_idx",
]

V15_4_BALANCE_4 = [
    "balance_neg_days",
    "balance_neg_days_ratio",
    "balance_min",
    "balance_mean",
]

V15_4_RISK_6 = [
    "risk_inflow_gap_max_norm",
    "risk_late_month_txn_ratio",
    "risk_velocity_txn_recent",
    "risk_velocity_outflow_recent",
    "risk_weekend_txn_ratio",
    "risk_overdraft_months_ratio",
]

FEATURE_SETS = {
    "v15_baseline_44": V15_BASELINE_44,
    "v15_4_fft_6": V15_4_FFT_6,
    "v15_4_risk_6": V15_4_RISK_6,
    "v15_4_balance_4": V15_4_BALANCE_4,
    "v15_4_keep_56": V15_BASELINE_44 + V15_4_FFT_6 + V15_4_RISK_6,
    "v15_4_keep_60": V15_BASELINE_44 + V15_4_FFT_6 + V15_4_RISK_6 + V15_4_BALANCE_4,
}
