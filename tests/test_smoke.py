"""Smoke tests for the edgeR negative-binomial DE workflow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyedger


def _make_counts(seed=0, n_genes=400, n_per_group=4):
    """Synthetic count matrix with planted DE genes."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    base = rng.gamma(2.0, 150.0, n_genes)
    is_de = rng.uniform(size=n_genes) < 0.15
    fc = np.where(is_de, rng.choice([0.25, 4.0], n_genes), 1.0)
    counts = np.zeros((n_genes, n))
    for g in (0, 1):
        mu = base * (fc if g else 1.0)
        counts[:, g * n_per_group:(g + 1) * n_per_group] = rng.poisson(
            mu[:, None], (n_genes, n_per_group))
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    design = np.zeros((n, 2)); design[:, 0] = 1.0; design[n_per_group:, 1] = 1.0
    return counts, group, design, is_de


def test_ql_workflow():
    counts, group, design, is_de = _make_counts()
    dge = pyedger.DGEList(counts=counts, group=group)
    dge = pyedger.calcNormFactors(dge)
    dge = pyedger.estimateDisp(dge, design=design)
    fit = pyedger.glmQLFit(dge, design)
    qlf = pyedger.glmQLFTest(fit, coef=1)
    res = pyedger.topTags(qlf, n=np.inf)
    assert len(res) == counts.shape[0]
    top = set(res.index[:int(is_de.sum())])
    true_de = set(np.flatnonzero(is_de))
    assert len(top & true_de) / max(is_de.sum(), 1) > 0.5


def test_exact_test():
    counts, group, design, is_de = _make_counts(seed=1)
    dge = pyedger.DGEList(counts=counts, group=group)
    dge = pyedger.calcNormFactors(dge)
    dge = pyedger.estimateDisp(dge, design=design)
    et = pyedger.exactTest(dge)
    res = pyedger.topTags(et, n=np.inf)
    assert len(res) == counts.shape[0]


def test_calc_norm_factors_and_filter():
    counts, group, design, _ = _make_counts(seed=2)
    dge = pyedger.DGEList(counts=counts, group=group)
    dge = pyedger.calcNormFactors(dge)
    nf = np.asarray(dge.samples["norm.factors"] if hasattr(dge, "samples")
                    else dge.norm_factors)
    assert np.all(np.isfinite(nf)) and np.all(nf > 0)
    keep = pyedger.filterByExpr(dge, group=group)
    assert keep.sum() > 0


def test_cpm():
    counts, group, _, _ = _make_counts(seed=3)
    dge = pyedger.DGEList(counts=counts, group=group)
    c = pyedger.cpm(dge)
    assert np.asarray(c).shape == counts.shape


def test_rstyle_aliases_exist():
    for nm in ("DGEList", "calcNormFactors", "estimateDisp", "glmFit",
               "glmLRT", "glmQLFit", "glmQLFTest", "exactTest", "topTags",
               "filterByExpr", "cpm"):
        assert hasattr(pyedger, nm), nm


def test_dgelist_subset_like_r():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.poisson(20, (50, 6)), index=[f"g{i}" for i in range(50)], columns=list("abcdef"))
    y = pyedger.DGEList(X, group=pd.Categorical(list("xxxyyy")))
    keep = pyedger.filterByExpr(y)
    kept = y[keep]                                             # R default: keep.lib.sizes = TRUE
    np.testing.assert_array_equal(kept.samples["lib.size"], y.samples["lib.size"])
    assert kept.counts.shape == (int(np.sum(keep)), 6)
    re = y.subset(keep, keep_lib_sizes=False)                  # y[keep, , keep.lib.sizes = FALSE]
    np.testing.assert_array_equal(re.samples["lib.size"], X.loc[np.asarray(keep)].sum(0).to_numpy(float))
    two = y[["g1", "g3"], ["a", "d"]]
    assert two.counts.shape == (2, 2) and list(two.samples["group"].cat.categories) == ["x", "y"]


def test_estimate_disp_positional_design_matches_keyword():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.negative_binomial(5, 0.2, (200, 6)))
    group = np.array(["a"] * 3 + ["b"] * 3)
    design = np.column_stack([np.ones(6), (group == "b").astype(float)])
    a = pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=group)), design)
    b = pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=group)), design=design)
    np.testing.assert_array_equal(a.tagwise_dispersion, b.tagwise_dispersion)
    with pytest.warns(FutureWarning):
        pyedger.estimateDisp(pyedger.calcNormFactors(pyedger.DGEList(X, group=group)), group)


def test_version():
    assert isinstance(pyedger.__version__, str) and pyedger.__version__


def test_threads_do_not_change_results(monkeypatch):
    """Row blocks are independent, so the thread count must not change any result bit."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.negative_binomial(3, 0.05, (5000, 8)))
    group = pd.Categorical(["a"] * 4 + ["b"] * 4)
    design = np.column_stack([np.ones(8), np.asarray(group) == "b"]).astype(float)

    def run():
        y = pyedger.estimateDisp(pyedger.normLibSizes(pyedger.DGEList(X, group=group)), design)
        q = pyedger.glmQLFTest(pyedger.glmQLFit(y, design), coef=1).table
        e = pyedger.exactTest(y).table
        return y.tagwise_dispersion, q["PValue"].to_numpy(), e["PValue"].to_numpy()

    monkeypatch.setenv("PYEDGER_NUM_THREADS", "1")
    one = run()
    monkeypatch.setenv("PYEDGER_NUM_THREADS", "4")
    four = run()
    for a, b in zip(one, four):
        np.testing.assert_array_equal(a, b)
