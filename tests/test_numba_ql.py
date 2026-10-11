"""The numba QL kernels agree with the NumPy path (ulp level)."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import pyedger
from pyedger import _glm_batch as gb
from pyedger import _numba_kernels as nk
from pyedger import _ql_adjust as qa

pytestmark = pytest.mark.skipif(not nk.HAVE_NUMBA, reason="numba not available")
DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def inputs():
    X = pd.read_csv(DATA / "counts.tsv", sep="\t", index_col=0)
    lib = X.columns.str.replace(r"_(KO|NT)$", "", regex=True)
    cond = pd.Categorical(X.columns.str.split("_").str[-1], ["NT", "KO"])
    D = pd.get_dummies(pd.DataFrame({"lib": pd.Categorical(lib), "cond": cond}), drop_first=True).astype(float)
    D.insert(0, "Intercept", 1.0)
    y = pyedger.normLibSizes(pyedger.DGEList(X, group=cond))
    fit = pyedger.glmFit(y, D.to_numpy(), dispersion=0.1)
    Y = X.to_numpy(float)
    rng = np.random.default_rng(1)
    disp = np.broadcast_to(rng.uniform(0.001, 3.9, Y.shape[0])[:, None], Y.shape)
    W = rng.uniform(0.5, 2.0, Y.shape)
    return Y, fit.fitted_values, D.to_numpy(), disp, W, pyedger.core.ave_log_cpm(y)


def _both(monkeypatch, f):
    a = f()
    monkeypatch.setattr(nk, "HAVE_NUMBA", False)
    b = f()
    monkeypatch.setattr(nk, "HAVE_NUMBA", True)
    return a, b


def test_adj_vec_mat(inputs, monkeypatch):
    Y, mu, X, disp, W, _ = inputs
    for fn in (qa.compute_adj_vec, qa.compute_adj_mat):
        a, b = _both(monkeypatch, lambda: fn(Y, mu, X, disp, 1.7, W))
        for k in a:
            # hat values of ill-conditioned weighted designs (zero fits) depend on the
            # summation order inside dqrdc2, as R's own do on the BLAS: ~eps * cond
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-10)


def test_ave_qd_and_one_group(inputs, monkeypatch):
    Y, mu, X, disp, W, A = inputs
    a, b = _both(monkeypatch, lambda: qa.compute_ave_qd(Y, mu, X, disp, A, W))
    np.testing.assert_allclose(a, b, rtol=1e-12)
    off = np.broadcast_to(np.log(Y.sum(0)), Y.shape)
    a, b = _both(monkeypatch, lambda: gb.mglm_one_group(Y, off, disp, W))
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_exact_weight_branch_falls_back(inputs):
    # phi >= 4.001 uses the exact finite sums, which only the NumPy path implements
    Y, mu, X, _, W, _ = inputs
    out = qa.compute_adj_vec(Y[:20], mu[:20], X, np.full(20, 5.0), 1.0, W[:20])
    assert np.all(np.isfinite(out["s2"]))
