# PWI Process Window Cost Optimization Review

## Purpose

This note summarizes review targets for improving the current PWI process window
analysis from a compute-cost perspective. The goal is to keep the current
statistical intent intact where possible, while reducing repeated data movement,
unnecessary pairwise tests, and full-row modeling cost.

The main pipeline is:

```text
preprocess -> assign_groups -> hypothesis test -> posthoc -> compute_window -> compute_pwi
```

Relevant implementation files:

- `pwi_analysis/preprocess.py`
- `pwi_analysis/grouping.py`
- `pwi_analysis/hypothesis.py`
- `pwi_analysis/posthoc.py`
- `pwi_analysis/windowing.py`
- `pwi_analysis/parallel.py`
- `pwi_analysis/pipeline.py`

## Current Cost Profile

### 1. Repeated filtering and merge across task combinations

`run_parallel_pwi` builds all `(m_key2, bin_id)` combinations, then each
`_process_one` filters `metro_all` and `eds_all` and calls the full pipeline.

```python
m_temp = metro_all[metro_all["m_key2"] == m_key]
e_temp = eds_all[eds_all["bin_id"] == bin_id]
result, message = pwi_analysis(m_temp, e_temp, cfg)
```

Inside `preprocess`, each task performs:

- inner merge on `root_lot_id`, `wafer_id`
- quantile calculation on `item_value`
- quantile calculation on `bin_value`
- boolean filtering and copy

If there are `K` metro keys and `B` bin IDs, the same wafer-key alignment is
effectively recomputed `K * B` times. For large manufacturing data, this can
dominate the statistical test cost.

Expected cost pattern:

- Repeated merge: roughly `O(K * B * join_rows)`
- Repeated copies: high memory bandwidth and process serialization overhead
- Joblib process mode may duplicate large DataFrames across workers

### 2. Group assignment sorts every task

`assign_groups` ranks `item_value` and then applies `qcut`.

```python
df["group"] = pd.qcut(
    df["item_value"].rank(method="first"),
    cfg.n_groups,
    labels=labels,
)
```

This is reasonable for a single analysis, but under many combinations the rank
or quantile assignment is repeated for every `(m_key2, bin_id)` pair. Because
groups depend only on `item_value` for a given `m_key2`, the same group labels
can be reused across multiple `bin_id` values.

Expected cost pattern:

- Per task ranking/sorting: roughly `O(n log n)`
- Potential reuse opportunity: compute once per `m_key2`, then join with each
  `bin_id`

### 3. One-way ANOVA uses formula OLS machinery

`_run_anova` uses `statsmodels.formula.api.ols` and then `anova_lm`.

```python
model = ols("bin_value ~ C(group)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

For one-way group comparison, this is convenient but heavier than necessary.
It builds a design matrix and model object even though the test can be computed
from group counts, means, and variances.

Expected cost pattern:

- Fine for small ad hoc analysis
- Avoidable overhead for many repeated tasks

### 4. Posthoc computes full pairwise matrix but only one row is used

`identify_good_groups` runs `posthoc_tamhane` or `posthoc_dunn`, then selects
only the row corresponding to the reference group.

```python
posthoc_df = _run_posthoc(df, is_normal)
group_means = df.groupby("group", observed=True)["bin_value"].mean()
ref_group = group_means.idxmin()
p_vals = posthoc_df.loc[ref_group]
```

The business definition is:

> Good groups are groups not significantly different from the lowest mean
> `bin_value` group.

That only requires comparisons against the reference group. A full `G x G`
posthoc matrix is unnecessary unless all pairwise relationships are later used.

Expected cost pattern:

- Current posthoc: approximately `O(G^2)` group comparisons
- Needed for current decision: approximately `O(G)`

With default `n_groups=10`, this is not the largest bottleneck. It becomes more
important if `n_groups` increases or the analysis runs across many combinations.

### 5. Window fitting uses all rows on each side

`compute_window` fits a quadratic separately to the left and right side of the
good region.

```python
coeffs = np.polyfit(x, y, 2)
roots = np.roots(shifted_coeffs)
```

A quadratic fit is cheap relative to the merge and sort, but it currently uses
all raw rows. The fit could often be done on binned summaries with weights,
especially when the data size per task is large.

Expected cost pattern:

- Current raw-row fit: roughly `O(n)` per side for degree-2 least squares
- Weighted binned fit: roughly `O(M)`, where `M` is fixed number of x bins

## Recommended Improvements

### Priority 1: Precompute aligned/pivoted data and avoid repeated merge

Build a reusable representation keyed by `root_lot_id`, `wafer_id`.

Possible shape:

- one row per wafer key
- metro columns widened by `m_key2`
- EDS columns widened by `bin_id`

Conceptual layout:

```text
root_lot_id | wafer_id | item_value:KEY_001 | item_value:KEY_002 | bin_value:BIN_001 | bin_value:BIN_002
```

Then each task can select two numeric columns instead of filtering and merging
two long DataFrames.

Expected benefit:

- Remove `K * B` repeated joins
- Reduce DataFrame copies
- Allow group labels to be cached by `m_key2`

Risks and questions:

- Duplicate `(root_lot_id, wafer_id, m_key2)` or `(root_lot_id, wafer_id, bin_id)`
  rows need a deterministic aggregation rule.
- Sparse combinations need careful NaN handling.
- Existing tests assume long input format, so this should be added as an
  internal execution strategy while preserving public API compatibility.

Suggested API direction:

```python
prepared = prepare_pwi_inputs(metro_all, eds_all, cfg)
results = run_parallel_pwi_prepared(prepared, cfg, n_jobs=-1)
```

Keep `run_parallel_pwi` as a wrapper for compatibility.

### Priority 2: Cache group labels per `m_key2`

Since group assignment depends on `item_value`, not `bin_value`, it can be
computed once for each metro key and reused across all bins.

Current repeated path:

```text
KEY_001 x BIN_001 -> assign_groups(KEY_001 item_value)
KEY_001 x BIN_002 -> assign_groups(KEY_001 item_value)
KEY_001 x BIN_003 -> assign_groups(KEY_001 item_value)
```

Proposed path:

```text
KEY_001 -> assign_groups once
KEY_001 group labels + each BIN column -> tests/windowing
```

Expected benefit:

- Avoid repeated rank/qcut cost
- Improves consistency across bin comparisons for the same process metric

Risk:

- Outlier filtering currently depends on both `item_value` and `bin_value`.
  If EDS outlier filtering removes different rows per `bin_id`, group counts
  can differ by bin. The cache design should decide whether to:
  - assign groups before EDS outlier filtering, then filter rows;
  - or cache item-value quantile cut points and apply them after bin filtering.

The second option preserves more of the current semantics.

### Priority 3: Replace full posthoc matrix with reference-only comparisons

The current good-group decision only compares each group with the reference
group, where the reference group has the lowest mean `bin_value`.

Recommended approach:

- Normal path: Welch-style pairwise comparison against reference, with multiple
  testing correction over `G - 1` comparisons.
- Non-normal path: Dunn-style or Mann-Whitney reference-vs-group comparisons,
  again corrected over `G - 1` comparisons.

Expected benefit:

- `O(G^2)` to `O(G)`
- Lower memory usage
- More direct match to the business rule

Statistical caution:

- The current method treats "not significantly different" as "good." This can
  be too lenient when sample size is small or power is low. A better definition
  may use equivalence testing.

### Priority 4: Consider equivalence testing for good group definition

Current interpretation:

```text
p >= alpha vs reference => good group
```

This means failure to detect a difference is treated as evidence of acceptable
similarity. Statistically, that is weaker than proving practical equivalence.

Alternative:

```text
group is good if it is statistically equivalent to the reference group within
a practical bin_value margin
```

Candidate method:

- TOST, two one-sided tests
- Margin based on domain tolerance, historical noise, or fraction of good-group
  standard deviation

Expected benefit:

- Better decision quality
- Still reference-only and computationally cheap

Risk:

- Requires process-domain agreement on equivalence margin.
- Output may be stricter than current method.

### Priority 5: Replace formula OLS ANOVA with summary-statistic tests

For one-way group comparison, compute from grouped summaries:

- count
- mean
- variance
- rank summaries for nonparametric path

Options:

- `scipy.stats.f_oneway` for standard one-way ANOVA
- Welch ANOVA using group summary statistics
- Kruskal-Wallis can remain in SciPy, but avoid repeated groupby materialization
  where possible

Expected benefit:

- Less object allocation
- Lower overhead in high-throughput runs

Risk:

- Results may differ slightly from `statsmodels` type-2 ANOVA in edge cases.
  For one-way categorical group designs this should usually be acceptable, but
  it needs regression tests.

### Priority 6: Fit process window on weighted summaries

Instead of fitting quadratic curves to all raw rows, summarize by x bins:

```text
item_value_bin_center | mean_bin_value | count
```

Then fit with weights:

```python
np.polyfit(x_bin_center, mean_bin_value, deg=2, w=np.sqrt(count))
```

Expected benefit:

- Raw rows per task no longer dominate window fitting
- More stable curve under very large data
- Easier to add robust smoothing

Risks:

- Binning can bias boundary estimates if bins are too coarse.
- Need tests comparing raw-row and weighted-summary windows on synthetic data.

### Priority 7: Use GAM/spline surrogate when combinations become very large

If the target workload has many `m_key2 x bin_id` combinations, a global model
can amortize cost:

```text
bin_value ~ smooth(item_value) + m_key2 + bin_id + interactions
```

Candidates:

- GAM with spline terms
- shape-constrained spline, if the response is expected to be U-shaped
- histogram-based gradient boosting as a fast surrogate, if interpretability
  constraints are lower

Recommended preference:

1. GAM or spline first, because it is easier to explain and validate.
2. GBDT only if predictive accuracy matters more than direct statistical
   interpretability.
3. Deep learning is not recommended for this workload unless the feature space
   grows far beyond the current two-axis setup.

Risks:

- Requires validation against current PWI outputs.
- A global surrogate changes the analysis definition more than the earlier
  optimizations.
- Model governance and explainability may matter in semiconductor process use.

## Suggested Implementation Sequence

1. Add benchmark fixtures that scale `samples`, `keys`, and `bins`.
2. Add timing around `preprocess`, `assign_groups`, `run_hypothesis_test`,
   `identify_good_groups`, and `compute_window`.
3. Implement prepared wide/pivot execution behind a new internal API.
4. Cache item-value quantile cut points or group labels per `m_key2`.
5. Replace full posthoc matrix with reference-only comparison.
6. Replace formula OLS with lighter one-way test implementation.
7. Add optional weighted-summary window fitting.
8. Only then evaluate GAM/spline surrogate.

## Benchmark Metrics

Track at minimum:

- wall-clock runtime by pipeline stage
- peak memory usage
- number of DataFrame copies
- result parity:
  - p-value category: significant vs non-significant
  - good group set similarity
  - window low/high absolute error
  - PWI percentage difference

Recommended synthetic workload grid:

```text
samples per key: 1k, 10k, 100k
m_key2 count:    5, 50, 200
bin_id count:    3, 20, 100
n_groups:        10, 20, 50
```

## Questions For Claude Code Review

Please review this branch with these questions:

1. Are the identified cost bottlenecks consistent with the actual code paths?
2. Which proposed optimization is likely to produce the largest runtime and
   memory improvement first?
3. Does reference-only posthoc preserve the current good-group business rule?
4. What edge cases would break a pivoted/prepared representation?
5. Is equivalence testing a better statistical framing for good-group detection?
6. Would weighted binned window fitting materially change window boundaries?
7. What tests should be added before changing implementation?

## Short Recommendation

Start with data-layout optimization, not AI modeling.

The highest-confidence path is:

```text
prepared wafer-key table
-> cached item_value grouping per m_key2
-> reference-only posthoc
-> lightweight one-way tests
-> optional weighted window fit
```

GAM/spline surrogate modeling is worth evaluating only after the repeated merge,
ranking, and posthoc costs are removed or measured.
