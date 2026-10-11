"""Numerical parity with Bioconductor edgeR 4.0.16.

Reference values are produced by tests/data/make_reference.R from a simulated
paired design (6 libraries x {NT, KO}) that includes all-zero library pairs,
so the zero-fit handling, reduced designs and residual-df corrections are
exercised. Tolerances are loose relative to the observed agreement (~1e-9).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import pyedger

DATA = pathlib.Path(__file__).parent / "data"
RTOL = 1e-6
# edgeR's dispCoxReid maximises with optimize(tol=1e-5) on the dispersion^(1/4)
# scale, so its common dispersion is only defined to ~4e-5 relative; values
# that depend on it (glmQLFit with legacy=False) are compared at that level.
RTOL_COXREID = 1e-4


def _read(name):
    return pd.read_csv(DATA / name, sep="\t")


@pytest.fixture(scope="module")
def setup():
    X = pd.read_csv(DATA / "counts.tsv", sep="\t", index_col=0)
    lib = X.columns.str.replace(r"_(KO|NT)$", "", regex=True)
    cond = pd.Categorical(X.columns.str.split("_").str[-1], ["NT", "KO"])
    y = pyedger.DGEList(X, group=cond)
    keep = np.asarray(pyedger.filterByExpr(y, group=np.asarray(cond)))
    np.testing.assert_array_equal(np.flatnonzero(keep), _read("keep.tsv")["keep"].to_numpy())
    y = pyedger.calcNormFactors(pyedger.DGEList(X[keep], group=cond))
    D = pd.get_dummies(pd.DataFrame({"lib": pd.Categorical(lib), "cond": cond}), drop_first=True).astype(float)
    D.insert(0, "Intercept", 1.0)
    return X[keep], cond, D


def _close(a, b, rtol=RTOL, atol=0.0):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=rtol, atol=atol)


def test_design_estimate_disp(setup):
    X, cond, D = setup
    y = pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=cond)), design=D.to_numpy())
    s = _read("design_disp_scalar.tsv")
    _close(y.common_dispersion, s["common"][0])
    _close(y.prior_df, s["prior_df"][0])
    r = _read("design_disp.tsv")
    _close(y.trended_dispersion, r["trended"])
    _close(y.tagwise_dispersion, r["tagwise"])
    _close(y.AveLogCPM, r["ave"])


@pytest.mark.parametrize("legacy", [True, False])
def test_design_ql(setup, legacy):
    X, cond, D = setup
    y = pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=cond)), design=D.to_numpy())
    fit = pyedger.glmQLFit(y, D, legacy=legacy)
    res = pyedger.glmQLFTest(fit, coef=D.shape[1] - 1).table
    r = _read(f"design_ql_legacy{int(legacy)}.tsv")
    rtol, atol = (RTOL, 1e-7) if legacy else (RTOL_COXREID, 1e-5)
    _close(res["logFC"], r["logFC"], rtol=rtol, atol=atol)
    _close(res["logCPM"], r["logCPM"])
    _close(res["F"], r["F"], rtol=rtol, atol=atol)
    _close(res["PValue"], r["PValue"], rtol=rtol, atol=atol)
    _close(fit.s2_post, r["s2post"], rtol=rtol)
    _close(np.broadcast_to(fit.df_prior, len(r)), r["dfprior"], rtol=rtol)


def test_classic_path(setup):
    X, cond, _ = setup
    y = pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=cond)))
    s = _read("classic_disp_scalar.tsv")
    _close(y.common_dispersion, s["common"][0])
    _close(y.prior_df, s["prior_df"][0])
    r = _read("classic.tsv")
    _close(y.trended_dispersion, r["trended"])
    _close(y.tagwise_dispersion, r["tagwise"])
    res = pyedger.glmQLFTest(pyedger.glmQLFit(y)).table
    _close(res["F"], r["F"], atol=1e-9)
    _close(res["PValue"], r["PValue"])
    et = pyedger.exactTest(y).table
    _close(et["logFC"], r["exact_logFC"], atol=1e-7)
    _close(et["PValue"], r["exact_PValue"])


@pytest.mark.parametrize("k,disp", [(1, 0.001), (2, 0.1), (3, 5.0)])
def test_glm_fit(setup, k, disp):
    X, cond, D = setup
    y = pyedger.DGEList(X, group=cond)
    y = pyedger.calcNormFactors(y)
    fit = pyedger.glmFit(y, D.to_numpy(), dispersion=disp, prior_count=0)
    r = _read(f"glmfit_disp{k}.tsv")
    _close(fit.deviance, r["deviance"], rtol=1e-6, atol=1e-8)
    coef = r.drop(columns="deviance").to_numpy()
    ok = np.abs(coef) < 15                     # coefficients heading to -inf (zero fits) are not identified
    np.testing.assert_allclose(fit.coefficients[ok], coef[ok], rtol=1e-5, atol=1e-6)


def test_glm_common_disp(setup):
    X, cond, D = setup
    y = pyedger.calcNormFactors(pyedger.DGEList(X, group=cond))
    y = pyedger.estimateGLMCommonDisp(y, D.to_numpy())
    _close(y.common_dispersion, _read("glm_common.tsv")["common"][0], rtol=RTOL_COXREID)


def test_tmm_tied_log_ratios():
    # Two pseudobulk ATAC libraries (GSE274113 IRF9 population, KO rep vs TMM
    # reference column) stored as unique (obs, ref) pairs with multiplicities.
    # The log-ratios are heavily tied, so an ulp difference between NumPy's
    # log2 and libm's moves ties across the trim boundary (0.66398 vs R).
    # R: p <- read.delim("tmm_ties.tsv.gz"); edgeR:::.calcFactorTMM(rep(p$obs, p$n), rep(p$ref, p$n))
    p = _read("tmm_ties.tsv.gz")
    obs, ref = np.repeat(p["obs"], p["n"]).to_numpy(float), np.repeat(p["ref"], p["n"]).to_numpy(float)
    f = pyedger.core._calc_factor_tmm(obs, ref, obs.sum(), ref.sum(), 0.3, 0.05, True, -1e10)
    _close(f, 0.6646329070011745, rtol=1e-12)


def test_infinite_prior_df():
    # Counts with no extra-Poisson variability between genes: squeezeVar gives
    # prior.df = Inf, and edgeR then leaves tagwise at the common dispersion
    # instead of running WLEB (prior.n > 1e6). Previously this produced NaN.
    # R: y <- estimateDisp(calcNormFactors(DGEList(x, group=g))); glmLRT(glmFit(y), coef=2)
    d = _read("no_extra_var.tsv")
    X = d[[f"s{i}" for i in range(6)]]
    y = pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=["treatment"] * 3 + ["control"] * 3)))
    assert np.isinf(y.prior_df)
    _close(y.tagwise_dispersion, d["tagwise"], rtol=1e-9)
    _close(pyedger.glmLRT(pyedger.glmFit(y), coef="treatment").table["PValue"], d["PValue"], rtol=1e-8)
