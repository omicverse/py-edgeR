"""Numerical parity with Bioconductor edgeR 4.10.5 (limma 3.68.5).

Reference values are produced by tests/data/make_reference.R from a simulated
paired design (6 libraries x {NT, KO}) that includes all-zero library pairs,
so the zero-fit handling, reduced designs and residual-df corrections are
exercised. Tolerances are loose relative to the observed agreement.
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
# scale, so values that depend on it are only defined to ~4e-5 relative.
RTOL_COXREID = 1e-4


def _read(name):
    return pd.read_csv(DATA / name, sep="\t", float_precision="round_trip")


def _close(a, b, rtol=RTOL, atol=0.0):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=rtol, atol=atol)


@pytest.fixture(scope="module")
def setup():
    X = pd.read_csv(DATA / "counts.tsv", sep="\t", index_col=0)
    lib = X.columns.str.replace(r"_(KO|NT)$", "", regex=True)
    cond = pd.Categorical(X.columns.str.split("_").str[-1], ["NT", "KO"])
    keep = np.asarray(pyedger.filterByExpr(pyedger.DGEList(X, group=cond), group=np.asarray(cond)))
    np.testing.assert_array_equal(np.flatnonzero(keep), _read("keep.tsv")["keep"].to_numpy())
    D = pd.get_dummies(pd.DataFrame({"lib": pd.Categorical(lib), "cond": cond}), drop_first=True).astype(float)
    D.insert(0, "Intercept", 1.0)
    return X[keep], cond, D


def _dge(X, cond):
    return pyedger.normLibSizes(pyedger.DGEList(X, group=cond))


def test_norm_factors(setup):
    X, cond, _ = setup
    _close(_dge(X, cond).samples["norm.factors"], _read("norm_factors.tsv")["norm"], rtol=1e-12)


@pytest.mark.parametrize("legacy_span", [False, True])
def test_design_estimate_disp(setup, legacy_span):
    X, cond, D = setup
    y = pyedger.estimateDisp(_dge(X, cond), design=D.to_numpy(), legacy_span=legacy_span)
    sfx = "_legacyspan" if legacy_span else ""
    s = _read(f"design_disp{sfx}_scalar.tsv")
    _close(y.common_dispersion, s["common"][0])
    _close(y.prior_df, s["prior_df"][0])
    _close(y.span, s["span"][0], rtol=1e-12)
    r = _read(f"design_disp{sfx}.tsv")
    _close(y.trended_dispersion, r["trended"])
    _close(y.tagwise_dispersion, r["tagwise"])
    if not legacy_span:
        _close(y.AveLogCPM, r["ave"])


def _check_ql(fit, res, name, rtol=RTOL, atol=1e-7, scalars=True):
    r = _read(f"{name}.tsv")
    _close(res["logFC"], r["logFC"], rtol=rtol, atol=atol)
    _close(res["logCPM"], r["logCPM"])
    _close(res["F"], r["F"], rtol=rtol, atol=atol)
    _close(res["PValue"], r["PValue"], rtol=rtol, atol=atol)
    _close(fit.s2_post, r["s2post"], rtol=rtol)
    _close(np.broadcast_to(fit.s2_prior, len(r)), r["s2prior"], rtol=rtol)
    _close(np.broadcast_to(fit.df_prior, len(r)), r["dfprior"], rtol=rtol)
    sc = DATA / f"{name}_scalar.tsv"
    if scalars and sc.exists():
        s = _read(sc.name)
        _close(fit.top_proportion, s["top_proportion"][0], rtol=1e-12)
        _close(np.ravel(fit.dispersion)[0], s["dispersion"][0], rtol=rtol)
        _close(fit.average_ql_dispersion, s["ave_ql"][0], rtol=rtol)


@pytest.mark.parametrize("legacy", [True, False])
def test_design_ql(setup, legacy):
    X, cond, D = setup
    y = pyedger.estimateDisp(_dge(X, cond), design=D.to_numpy())
    fit = pyedger.glmQLFit(y, D, legacy=legacy)
    # legacy=False estimates the NB dispersion by Cox-Reid on the top genes (R's
    # optimize(tol=1e-5)), so it agrees with R only to the Cox-Reid level; see
    # test_design_ql_given_dispersion for the exact check of the rest.
    rtol, atol = (RTOL, 1e-7) if legacy else (RTOL_COXREID, 1e-5)
    _check_ql(fit, pyedger.glmQLFTest(fit, coef=D.shape[1] - 1).table, f"design_ql_legacy{int(legacy)}", rtol, atol)


def test_design_ql_given_dispersion(setup):
    # New QL method with R's Cox-Reid dispersion supplied: everything downstream
    # (average QL dispersion, adjusted deviance/df, squeezeVar, F-test) is exact.
    X, cond, D = setup
    y = pyedger.estimateDisp(_dge(X, cond), design=D.to_numpy())
    disp = _read("design_ql_legacy0_scalar.tsv")["dispersion"][0]
    fit = pyedger.glmQLFit(y, D, dispersion=disp)
    _check_ql(fit, pyedger.glmQLFTest(fit, coef=D.shape[1] - 1).table, "design_ql_legacy0", RTOL, 1e-9, scalars=False)
    assert fit.top_proportion is None             # edgeR leaves top.proportion NULL when dispersion is given


@pytest.mark.parametrize("legacy", [True, False])
def test_design_ql_robust(setup, legacy):
    X, cond, D = setup
    y = pyedger.estimateDisp(_dge(X, cond), design=D.to_numpy())
    disp = None if legacy else _read("design_ql_legacy0_scalar.tsv")["dispersion"][0]
    fit = pyedger.glmQLFit(y, D, legacy=legacy, robust=True, dispersion=disp)
    _check_ql(fit, pyedger.glmQLFTest(fit, coef=D.shape[1] - 1).table, f"design_ql_robust_legacy{int(legacy)}",
              RTOL, 1e-9, scalars=False)


def test_design_ql_without_dispersion(setup):
    # edgeR >= 4.10: the new QL method gives the same result whether or not the
    # DGEList carries dispersion estimates.
    X, cond, D = setup
    fit = pyedger.glmQLFit(_dge(X, cond), D)
    _check_ql(fit, pyedger.glmQLFTest(fit, coef=D.shape[1] - 1).table, "design_ql_nodisp", RTOL_COXREID, 1e-5)


def test_design_ql_options(setup):
    X, cond, D = setup
    y = pyedger.estimateDisp(_dge(X, cond), design=D.to_numpy())
    fit = pyedger.glmQLFit(y, D, top_proportion=0.2, abundance_trend=False)
    _check_ql(fit, pyedger.glmQLFTest(fit, coef=D.shape[1] - 1).table, "design_ql_top02_notrend")


def test_classic_path(setup):
    X, cond, _ = setup
    y = pyedger.estimateDisp(_dge(X, cond))
    s = _read("classic_disp_scalar.tsv")
    _close(y.common_dispersion, s["common"][0])
    _close(y.prior_df, s["prior_df"][0])
    r = _read("classic.tsv")
    _close(y.trended_dispersion, r["trended"])
    _close(y.tagwise_dispersion, r["tagwise"])
    res = pyedger.glmQLFTest(pyedger.glmQLFit(y, legacy=True)).table
    _close(res["F"], r["F_legacy"], atol=1e-9)
    _close(res["PValue"], r["PValue_legacy"])
    res = pyedger.glmQLFTest(pyedger.glmQLFit(y)).table
    _close(res["F"], r["F"], atol=1e-9)
    _close(res["PValue"], r["PValue"])
    et = pyedger.exactTest(y).table
    _close(et["logFC"], r["exact_logFC"], atol=1e-7)
    _close(et["PValue"], r["exact_PValue"])


@pytest.mark.parametrize("k,disp", [(1, 0.001), (2, 0.1), (3, 5.0)])
def test_glm_fit(setup, k, disp):
    X, cond, D = setup
    fit = pyedger.glmFit(_dge(X, cond), D.to_numpy(), dispersion=disp, prior_count=0)
    r = _read(f"glmfit_disp{k}.tsv")
    _close(fit.deviance, r["deviance"], rtol=1e-6, atol=1e-8)
    coef = r.drop(columns="deviance").to_numpy()
    ok = np.abs(coef) < 15                     # coefficients heading to -inf (zero fits) are not identified
    np.testing.assert_allclose(fit.coefficients[ok], coef[ok], rtol=1e-5, atol=1e-6)


def test_glm_common_disp(setup):
    X, cond, D = setup
    y = pyedger.estimateGLMCommonDisp(_dge(X, cond), D.to_numpy())
    _close(y.common_dispersion, _read("glm_common.tsv")["common"][0], rtol=RTOL_COXREID)


def test_milo_configuration():
    # omicverse Milo.da_nhoods: TMM off, trend.method="none", new QL.
    Y = pd.read_csv(DATA / "milo_counts.tsv", sep="\t", index_col=0)
    X = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]]).astype(float)
    dge = pyedger.DGEList(counts=Y, norm_factors=np.ones(8), group=np.zeros(8))
    pyedger.estimateDisp(dge, design=X, trend_method="none", robust=False)
    s = _read("milo_disp_scalar.tsv")
    _close(dge.common_dispersion, s["common"][0])
    assert np.isinf(dge.prior_df) == np.isinf(s["prior_df"][0])
    _close(dge.tagwise_dispersion, _read("milo_disp.tsv")["tagwise"])
    fit = pyedger.glmQLFit(dge, design=X, robust=False, legacy=False)
    _check_ql(fit, pyedger.glmQLFTest(fit, coef=1).table, "milo_ql")


def test_tmm_tied_log_ratios():
    # Two pseudobulk ATAC libraries (GSE274113 IRF9 population, KO rep vs TMM
    # reference column) stored as unique (obs, ref) pairs with multiplicities.
    # The log-ratios are heavily tied, so an ulp difference between NumPy's
    # log2 and libm's moves ties across the trim boundary (0.66398 vs R).
    p = _read("tmm_ties.tsv.gz")
    obs, ref = np.repeat(p["obs"], p["n"]).to_numpy(float), np.repeat(p["ref"], p["n"]).to_numpy(float)
    f = pyedger.core._calc_factor_tmm(obs, ref, obs.sum(), ref.sum(), 0.3, 0.05, True, -1e10)
    _close(f, _read("tmm_ties_ref.tsv")["factor"][0], rtol=1e-12)


def test_infinite_prior_df():
    # No extra-Poisson variability between genes: squeezeVar gives prior.df =
    # Inf, and edgeR then leaves tagwise at the trend instead of running WLEB
    # (prior.n > 1e6). pyedger <= 0.1.2 produced NaN here.
    d = _read("no_extra_var.tsv")
    r = _read("no_extra_var_ref.tsv")
    X = d[[f"s{i}" for i in range(6)]]
    y = pyedger.estimateDisp(_dge(X, ["treatment"] * 3 + ["control"] * 3))
    assert np.isinf(y.prior_df) and np.isinf(r["prior_df"][0])
    _close(y.tagwise_dispersion, r["tagwise"], rtol=1e-9)
    _close(pyedger.glmLRT(pyedger.glmFit(y), coef="treatment").table["PValue"], r["PValue"], rtol=1e-8)


def test_limma_hyperparameters():
    # limma 3.68: fitFDistRobustly (equal / unequal df1, covariate) and
    # squeezeVar choosing fitFDistUnequalDF1 for unequal df (robust, covariate).
    from pyedger import _limma
    i, r = _read("limma_in.tsv"), _read("limma_out.tsv")
    x, cov, du = i["x"].to_numpy(), i["cov"].to_numpy(), i["df1u"].to_numpy(float)
    for k, fit in [(1, _limma.fit_f_dist_robustly(x, 4.0)), (2, _limma.fit_f_dist_robustly(x, 4.0, covariate=cov)),
                   (3, _limma.fit_f_dist_robustly(x, du))]:
        _close(np.broadcast_to(fit["scale"], len(r)), r[f"s{k}"], rtol=1e-9)
        _close(fit["df2_shrunk"], r[f"d{k}"], rtol=1e-9)
    for k, kw in [(4, dict(robust=True)), (5, dict(covariate=cov, robust=True)), (6, dict(covariate=cov))]:
        dp, _, post = _limma.squeeze_var(x, du, **kw)
        _close(post, r[f"p{k}"], rtol=1e-9)
        _close(np.broadcast_to(dp, len(r)), r[f"d{k}"], rtol=1e-9)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_glm_treat(setup, k):
    X, cond, D = setup
    y = pyedger.estimateDisp(_dge(X, cond), design=D.to_numpy())
    fit = {1: lambda: pyedger.glmFit(y, D), 2: lambda: pyedger.glmQLFit(y, D, legacy=True), 3: lambda: pyedger.glmQLFit(y, D)}[k]()
    tab = pyedger.glmTreat(fit, coef=D.shape[1] - 1, lfc=np.log2(1.5)).table
    r = _read(f"treat{k}.tsv")
    # k=3: the new QL method estimates its own Cox-Reid dispersion (RTOL_COXREID)
    rtol, atol = (RTOL_COXREID, 1e-5) if k == 3 else (RTOL, 1e-7)
    _close(tab["logFC"], r["logFC"], rtol=rtol, atol=atol)
    _close(tab["PValue"], r["PValue"], rtol=rtol, atol=1e-12)
