# Feature 说明文档

各实验版本中使用的所有特征，含详细说明与交易类别映射。

## 交易类别编号对照表

| 编号 | 英文名 | 中文含义 |
|------|--------|---------|
| 0 | SELF_TRANSFER | 账户自转 |
| 1 | EXTERNAL_TRANSFER | 外部转账 |
| 2 | DEPOSIT | 存款 |
| 3 | PAYCHECK | 工资/薪酬 |
| 4 | MISCELLANEOUS | 杂项 |
| 5 | PAYCHECK_PLACEHOLDER | 薪酬占位符 |
| 6 | REFUND | 退款 |
| 7 | INVESTMENT_INCOME | 投资收益 |
| 8 | OTHER_BENEFITS | 其他福利 |
| 9 | UNEMPLOYMENT_BENEFITS | 失业救济金 |
| 10 | SMALL_DOLLAR_ADVANCE | 小额预支（发薪日贷款类） |
| 11 | TAX | 税款 |
| 12 | LOAN | 贷款 |
| 13 | INSURANCE | 保险 |
| 14 | FOOD_AND_BEVERAGES | 餐饮 |
| 15 | UNCATEGORIZED | 未分类 |
| 16 | GENERAL_MERCHANDISE | 一般商品零售 |
| 17 | AUTOMOTIVE | 汽车相关 |
| 18 | GROCERIES | 超市/杂货 |
| 19 | ATM_CASH | ATM 取现 |
| 20 | ENTERTAINMENT | 娱乐 |
| 21 | TRAVEL | 出行/旅游 |
| 22 | ESSENTIAL_SERVICES | 基础服务 |
| 23 | ACCOUNT_FEES | 账户手续费 |
| 24 | HOME_IMPROVEMENT | 家居装修 |
| 25 | OVERDRAFT | 透支 |
| 26 | CREDIT_CARD_PAYMENT | 信用卡还款 |
| 27 | HEALTHCARE_MEDICAL | 医疗健康 |
| 28 | PETS | 宠物 |
| 29 | EDUCATION | 教育 |
| 30 | GIFTS_DONATIONS | 礼品/捐款 |
| 31 | BILLS_UTILITIES | 账单/水电费 |
| 32 | MORTGAGE | 房贷 |
| 33 | CHILD_DEPENDENTS | 子女/抚养支出 |
| 34 | RENT | 租金 |
| 35 | BNPL | 先买后付（BNPL） |
| 36 | AUTO_LOAN | 车贷 |

---

## v15_baseline_44

v15 引入的基准特征集，共 44 个特征。

### 余额与交易金额

| 特征 | 说明 |
|------|------|
| `total_balance` | 评估日账户余额 |
| `amount_mean` | 全历史交易金额均值（有符号：流入为正，流出为负） |
| `amount_std` | 全历史交易金额标准差 |
| `amount_abs_mean` | 全历史交易绝对金额均值（不区分方向的平均笔金额） |
| `amount_abs_max` | 全历史最大单笔交易绝对金额 |

### 流入汇总

| 特征 | 说明 |
|------|------|
| `inflow_sum` | 全历史总流入金额 |
| `inflow_sum_7d` | 评估日前 7 天总流入金额 |
| `inflow_sum_30d` | 评估日前 30 天总流入金额 |

### 交易笔数

| 特征 | 说明 |
|------|------|
| `txn_count_7d` | 评估日前 7 天交易笔数 |

### 比率特征

| 特征 | 说明 |
|------|------|
| `inflow_to_balance_ratio` | `inflow_sum / abs(total_balance + 1)`，历史累计流入与当前余额的比值，反映资金周转规模 |
| `outflow_to_balance_ratio` | `outflow_sum / abs(total_balance + 1)`，历史累计流出与余额的比值 |
| `outflow_to_inflow_ratio` | `outflow_sum / inflow_sum`，整体支出率 |
| `recent_7d_to_30d_txn_ratio` | `txn_count_7d / txn_count_30d`，近 7 天交易是否集中于近 30 天 |
| `recent_30d_to_90d_txn_ratio` | `txn_count_30d / txn_count_90d`，近 30 天交易是否集中于近 90 天 |

### 类别特征 — 全历史

| 特征 | 类别 | 说明 |
|------|------|------|
| `all_cat_3_amount` | 工资 (3) | 全历史工资流入总额 |
| `all_cat_3_count` | 工资 (3) | 全历史工资交易笔数 |
| `all_cat_12_amount` | 贷款 (12) | 全历史贷款相关交易总额 |
| `all_cat_26_amount` | 信用卡还款 (26) | 全历史信用卡还款总额 |
| `all_cat_26_count` | 信用卡还款 (26) | 全历史信用卡还款笔数 |

### 类别特征 — 近 30 天

| 特征 | 类别 | 说明 |
|------|------|------|
| `d30_cat_3_amount` | 工资 (3) | 近 30 天工资流入总额 |
| `d30_cat_12_amount` | 贷款 (12) | 近 30 天贷款相关交易总额 |
| `d30_cat_23_amount` | 账户手续费 (23) | 近 30 天账户手续费支出总额 |
| `d30_cat_26_amount` | 信用卡还款 (26) | 近 30 天信用卡还款总额 |

### 类别特征 — 近 90 天

| 特征 | 类别 | 说明 |
|------|------|------|
| `d90_cat_3_count` | 工资 (3) | 近 90 天工资交易笔数 |
| `d90_cat_10_count` | 小额预支 (10) | 近 90 天小额预支（发薪日贷款类）笔数，信贷压力信号 |
| `d90_cat_12_amount` | 贷款 (12) | 近 90 天贷款相关交易总额 |
| `d90_cat_23_amount` | 账户手续费 (23) | 近 90 天账户手续费支出总额 |
| `d90_cat_26_amount` | 信用卡还款 (26) | 近 90 天信用卡还款总额 |

### 信用卡与零售特征

| 特征 | 说明 |
|------|------|
| `cc_payment_amount_share` | 信用卡还款金额（类别 26）占总流出的比例，值高说明大部分支出用于还卡 |
| `gen_merch_avg_txn_amount` | 一般商品零售（类别 16）的平均单笔交易金额，反映可自由支配消费水平 |

### 流入时序规律

| 特征 | 说明 |
|------|------|
| `inflow_gap_mean` | 相邻两笔流入之间的平均天数 |
| `inflow_gap_cv` | 流入间隔的变异系数（`std / mean`），值高 = 收入不规律 |
| `inflow_monthly_min` | 历史各月流入的最小值，反映收入下限 |

### 账户历史

| 特征 | 说明 |
|------|------|
| `history_days` | 首笔交易至评估日的天数 |
| `recency_days` | 最近一笔交易至评估日的天数，值高 = 账户近期不活跃 |

### 时间序列趋势特征（月度序列）

| 特征 | 说明 |
|------|------|
| `ts_inflow_slope` | 月度流入的 OLS 线性回归斜率（每月变化的绝对金额） |
| `ts_inflow_r2` | 月度流入线性拟合的 R²，反映趋势的解释力 |
| `ts_outflow_slope` | 月度流出的 OLS 斜率 |
| `ts_outflow_slope_norm` | 流出斜率除以月均流出（归一化） |
| `ts_net_slope` | 月度净现金流（流入 − 流出）的 OLS 斜率 |
| `ts_net_slope_norm` | 净现金流斜率除以月均净现金流（归一化） |
| `ts_inflow_d2_mean` | 月度流入的二阶差分均值，衡量收入趋势的加速/减速 |
| `ts_outflow_d2_mean` | 月度流出的二阶差分均值 |
| `ts_net_d2_mean` | 月度净现金流的二阶差分均值 |

---

## v15_4_fft_6

v15.4 新增的 6 个 FFT/频谱特征，从 15 个候选中筛选保留。基于零均值月度序列通过 `numpy.fft.rfft` 计算。

| 特征 | 序列 | 说明 |
|------|------|------|
| `fft_inflow_dom_power_ratio` | 流入 | 主导频率幅度 / 总 AC 幅度，值高 = 流入具有强周期性（如固定工资） |
| `fft_inflow_dom_freq_idx` | 流入 | 主导谐波的频率索引（1 = 全历史一个周期，2 = 两个周期，以此类推） |
| `fft_inflow_low_freq_ratio` | 流入 | 频率索引 1–2 的功率占总 AC 功率比例，值高 = 流入以慢变趋势为主 |
| `fft_inflow_spectral_entropy` | 流入 | 幅度谱的 Shannon 熵，低 = 周期性集中，高 = 噪声/不规律 |
| `fft_outflow_dom_freq_idx` | 流出 | 流出序列的主导谐波频率索引 |
| `fft_outflow_low_freq_ratio` | 流出 | 流出序列的低频功率占比 |

---

## v15_4_risk_6

v15.4 新增的 6 个信贷风险行为特征，从 16 个候选中筛选保留。参考 FPD / 早期预警研究。

| 特征 | 说明 |
|------|------|
| `risk_inflow_gap_max_norm` | `最长流入断档天数 / history_days`，收入中断时长相对账户历史的比例，值高 = 曾有长期无收入 |
| `risk_late_month_txn_ratio` | 发生在每月 25–31 日的交易占比，值高 = 月末财务压力模式 |
| `risk_velocity_txn_recent` | `(近30天日均交易笔数) / (历史日均交易笔数)`，近期交易频率是否相对历史基线加速 |
| `risk_velocity_outflow_recent` | `(近30天日均流出) / (历史日均流出)`，近期支出速率是否相对历史基线激增 |
| `risk_weekend_txn_ratio` | 发生在周六/周日的交易占总交易的比例，行为模式信号 |
| `risk_overdraft_months_ratio` | 历史各月中流出超过流入的月份占比，持续负现金流指标 |

---

## 特征集汇总

| 集合名称 | 特征数 | 说明 |
|---------|--------|------|
| `v15_baseline_44` | 44 | v15 基准特征集 |
| `v15_4_fft_6` | 6 | v15.4 新增 FFT/频谱特征 |
| `v15_4_risk_6` | 6 | v15.4 新增信贷风险行为特征 |
| `v15_4_keep_56` | 56 | 以上三个集合的并集（`v15_baseline_44 + v15_4_fft_6 + v15_4_risk_6`） |
