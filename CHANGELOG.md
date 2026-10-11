# Changelog

## 0.2.1

- **Speed:** optional numba kernels (`pip install pyedger[fast]`) for the Levenberg GLM fits, the adjusted profile likelihood, the new-QL adjustments (`compute_adj_vec` / `compute_adj_mat` / `compute_ave_qd`), the QL prior lowess and one-group fits. They follow edgeR's C code gene by gene and run in parallel over genes (threads: `NUMBA_NUM_THREADS`). On 167k features x 26 samples with a 14-column design the full `estimateDisp` + `glmQLFit` pipeline went from ~5 min to ~1 min (8 threads); edgeR 4.10.5 takes ~8 min single-threaded. Without numba (or with `PYEDGER_NO_NUMBA=1`) the NumPy implementation is used. Results agree with the NumPy path to ~1e-15 per kernel and with R as before (identical FDR calls on the GSE274113 checks).
- Removed an unused statsmodels import (statsmodels was never a declared dependency).

## 0.2.0

pyedger now targets **edgeR 4.10.5 / limma 3.68.5** (Bioconductor 3.23) instead of edgeR 4.0.16. Results change wherever edgeR changed between those releases; parity tests were regenerated with R 4.5.3.

Breaking defaults (as in edgeR >= 4.2/4.8):
- `glmQLFit(legacy=False)` is the default: the new QL method (Chen, Chen & Smyth 2024) as rewritten in edgeR 4.4+ (`compute_ave_qd` / `compute_adj_vec`; data-driven `top_proportion`; NB dispersion estimated on the top genes; dispersions stored in the DGEList are ignored; preset dispersions capped at 4). `legacy=True` gives the classic method.
- `estimateDisp` chooses its default span with `chooseLowessSpan(n)`; `legacy_span=True` restores the edgeR <= 4.6 span.

Other changes:
- limma 3.68 hyperparameter estimation: `squeezeVar` uses `fitFDistUnequalDF1` when residual df are unequal (e.g. zero fits), which changes `estimateDisp`'s `prior.df`; exact ports of `fitFDistRobustly` (replacing an approximation), `weightedLowess`/`loessFit`, `logmdigamma`, and R's `uniroot`.
- `glmQLFit(robust=True)` is supported; new `covariate_trend`, `keep_unit_mat`, `winsor_tail_p` arguments.
- `glmTreat`: `lfc=0` dispatches to `glmQLFTest` for QL fits; QL p-values respect the Poisson bound.
- Unit NB deviances are floored at 0 (edgeR 4.4+).
- `normLibSizes` alias for `calcNormFactors`.

## 0.1.3

- **`estimateDisp` with infinite prior df:** when the genewise variances show no extra variability, `squeezeVar` returns `prior.df = Inf`. edgeR then skips the weighted-likelihood step (prior.n > 1e6) and leaves the tagwise dispersions at the trend (or common) value; pyedger fed the infinite weight into it and returned NaN tagwise dispersions, so `glmFit`/`glmLRT` gave NaN for every gene. Ported edgeR's handling; regression test `test_infinite_prior_df`. (0.1.0 was not affected because it fixed prior df at 10.)

## 0.1.2

- **TMM parity fix:** `calcNormFactors(method="TMM")` now takes log2 through the C library, as R does. NumPy's SIMD `log2` differs by an ulp on some inputs, and TMM trims on ranks of heavily tied log-ratios, so the ulp moved ties across the trim boundary. On real pseudobulk ATAC data this shifted a norm factor by ~1e-3 (0.66398 vs R's 0.66463) and with it every downstream statistic; with the fix the 19 GSE274113 population comparisons match edgeR 4.0.16 to ~1e-8 in P-value with identical FDR calls. Regression test: `test_tmm_tied_log_ratios`.

## 0.1.1

Performance and numerical parity with edgeR 4.0.16.

- **Speed:** GLM fits are vectorised over genes (edgeR's Levenberg algorithm and one-way shortcut), replacing per-gene SciPy BFGS. `estimateDisp` with a design on ~1e5 features went from hours to about a minute.
- **Parity fixes** (each checked against R; see README):
  - `estimateDisp` with a design: zero-fitted observations dropped per zero pattern with reduced designs, warm-started grid fits, Cox-Reid adjustment with edgeR's LDL floor, AveLogCPM at the common dispersion, WLEB span, and the spline `maximizeInterpolant`.
  - `estimateDisp` classic mode: single Brent optimisation from pseudo-counts at dispersion 0.01, as edgeR does (previously `estimateCommonDisp`).
  - Prior df: residual df adjusted for exact zeros (`.residDF`); `prior_df=None` now estimates it, as in edgeR (was fixed at 10).
  - `squeezeVar`: the covariate (abundance trend) is now used; this is limma's natural-spline `fitFDist`.
  - `locfitByCol`: reproduces the locfit package (adaptive tree, vertex fits, interpolation) instead of a direct local regression.
  - `aveLogCPM`: one-group NB fit with prior counts (was a plain mean of log-CPM).
  - `glmQLFit(legacy=False)`: edgeR's bias-adjusted deviance and df (`compute_adjust_s2`) and `.computePriorS2`; the common dispersion is estimated on the top 5% of AveLogCPM. `abundance_trend` now defaults to `True`.
  - `glmLRT`/`glmQLFTest`: the null model uses the working dispersion for the new QL method; the Poisson bound is applied for legacy fits.
  - `estimateGLMCommonDisp`: Cox-Reid APL maximised with R's Brent `optimize`.
  - `q2qnbinom` in log space; `mglmOneGroup` as in edgeR; `exactTest` keeps negative pseudo-counts as R does and orders groups by factor level.
  - `DGEList` keeps a categorical `group`'s level order. A missing `group` means a single group, as in R.
  - `topTags(sort_by="none")` is supported.
- Tests: `tests/test_r_parity.py` with R reference values (`tests/data/make_reference.R`).

## 0.1.0

Initial release.
