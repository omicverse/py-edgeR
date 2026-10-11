# Changelog

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
