# Methodology

## Prediction task and representation

The pipeline predicts the binary `FPF_TARGET` label from a consumer's recorded
balance and transaction history. The first three characters of
`masked_consumer_id` identify clients C01, C02, and C03. The identifier supplies
group labels for training and evaluation; it is not a numerical model feature.
Transactions after a consumer's `evaluation_date` are excluded from feature
construction.

`cashflow_56` selects the same ordered inputs as the historical
`v15_4_keep_56` bundle:

| Family | Count | Examples |
|---|---:|---|
| Cashflow and trends | 44 | Balance, recent inflows, category totals, income gaps, monthly slopes |
| Frequency domain | 6 | Dominant-frequency index, low-frequency amplitude ratios, spectral entropy |
| Behavioral risk | 6 | Long income gaps, transaction velocity, late-month activity, overdraft months |

The implementation may compute supporting aggregates and additional candidates;
only the selected 56 columns reach the model. Four reconstructed-balance features
form a separate optional extension. See [the feature reference](feature_explanation.md)
and `features/sets.py` for exact definitions and selection.

## Router and experts

For a feature vector `x`, the multiclass router estimates `r_c(x)`, the probability
of membership in client `c`. Each expert estimates the positive-label probability
`p_c(x)` from examples in that client's training split. Prediction is:

```text
score(x) = sum(r_c(x) * p_c(x)) / sum(r_c(x))
           over fitted experts     over fitted experts
```

With all three experts present the denominator is one, up to numerical precision.
If a group has too few training rows or only one label, its expert is skipped and
the remaining router weights are renormalized. The router is trained on the
original training rows, before expert-specific resampling. Each expert uses its
own YAML-defined LightGBM parameters.

This is supervised client routing: the router learns client labels, not a joint
end-to-end gating objective. At inference it computes weights from features;
prediction is not a hard dispatch using the consumer ID. These distinctions
matter when explaining the model in an interview.

## Imbalance handling

The reference configuration keeps 20% of negatives for C01 and C03 and augments
their positives to twice the original positive count. Synthetic positive rows
interpolate between a positive example and one of its nearest positive neighbors
after constant-zero imputation. The effective neighbor count decreases when
fewer positive examples are available. All resampling is confined to training.
C02 uses its original training rows.

The refactor preserves the historical `sample_weight: true` behavior: every
expert receives weight `1 / negative_keep_ratio` for a negative row and
`1 / positive_multiplier` for a positive row. In the reference configuration
these weights are 5 and 0.5, including for C02. **For C02 these are class weights,
not compensation for a sampling operation**, because C02 is not resampled.
Changing this policy requires a separately named experiment and comparison;
the structural refactor deliberately does not change it.

## Evaluation and feature contributions

The split is stratified within each client: approximately 60% train, 20%
validation, and 20% test, with integer rounding for finite group sizes. It is a
random consumer split, not an out-of-time validation protocol. The main metric is
the unweighted mean of ROC-AUC across evaluable clients; pooled AUC and individual
client AUCs provide additional context. A group with only one observed label has
no defined AUC and is excluded from the group mean.
If no client in the holdout has both labels, evaluation fails with an explicit
error instead of writing an undefined AUC into the results JSON.

Permutation importance measures the decrease in mean client AUC after randomly
shuffling one feature across the evaluation rows. Repeated shuffles yield a mean
and standard deviation. Correlated features can substitute for one another, and
a negative importance can occur; this measures predictive dependence in this
model, not causation or an independent business effect.

The new `cashflow_moe_importance.yaml` configuration uses validation data for
importance. Historical configurations without an explicit `permutation_split`
retain their test-set analysis behavior. Using test importance repeatedly to
select features compromises an untouched final evaluation; perform new feature
selection on validation data and reserve test results for the final comparison.

## What the reorganization preserves

- Feature formulas, selected feature ordering, client routing, expert parameters,
  split seed, and resampling policy remain the reference algorithm.
- Monthly trend and FFT calculations use the observed active-month sequence.
  Missing calendar months are not newly inserted as zero-valued months. Changing
  that would change the frequency-domain interpretation and belongs in an
  explicit experiment.
- The 7/30/90-day windows include their boundary day. Feature-level missingness
  is retained until model imputation, and the FFT ratios use spectral amplitudes
  rather than squared power despite their historical column names.
- The historical balance configurations disable sample weighting, whereas the
  56-feature reference enables it. Their results therefore do not isolate the
  value of balance features. A clean ablation must hold training policy fixed.
- The original data and verified real-data scores are not provided.
  Demo runs demonstrate reproducibility and execution, not real-world credit
  performance, calibration, fairness, or superiority over a single model.
