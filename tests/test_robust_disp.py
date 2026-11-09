"""estimate_glm_robust_disp / glm_residuals (edgeR estimateGLMRobustDisp)."""
import numpy as np
import pandas as pd

import pyedger as pe


def _data(seed=1, G=400, n=8, n_out=20):
    rs = np.random.RandomState(seed)
    mu = rs.lognormal(4, 1.2, G)[:, None] * np.ones((1, n))
    disp = 0.1
    Y = rs.negative_binomial(1 / disp, 1 / (1 + disp * mu)).astype(float)
    out = rs.choice(G, n_out, replace=False)
    col = rs.randint(0, n, n_out)
    Y[out, col] *= 20
    counts = pd.DataFrame(Y, index=[f"g{i}" for i in range(G)], columns=[f"s{j}" for j in range(n)])
    design = pd.DataFrame({"Intercept": 1.0, "cond": [0] * (n // 2) + [1] * (n - n // 2)},
                          index=counts.columns)
    return counts, design, out, col


def test_glm_residuals_pearson_and_deviance():
    counts, design, _, _ = _data()
    y = pe.DGEList(counts, group=["a"] * 4 + ["b"] * 4)
    pe.estimate_disp(y, design=design)
    fit = pe.glm_fit(y, design=design)
    r = pe.glm_residuals(fit, type="pearson")
    mu = fit.fitted_values
    phi = np.asarray(fit.dispersion).reshape(-1, 1)
    np.testing.assert_allclose(r, (counts.to_numpy() - mu) / np.sqrt(mu + phi * mu * mu), rtol=1e-12)
    d = pe.glm_residuals(fit, type="deviance")
    assert d.shape == r.shape and np.all(np.sign(d) == np.sign(counts.to_numpy() - mu))


def test_robust_disp_downweights_outliers():
    counts, design, out, col = _data()
    y = pe.DGEList(counts, group=["a"] * 4 + ["b"] * 4)
    y = pe.estimateGLMRobustDisp(y, design=design)
    w = y.weights.to_numpy()
    assert np.all(w[out, col] < 0.5)
    mask = np.ones_like(w, dtype=bool)
    mask[out, col] = False
    assert np.median(w[mask]) == 1.0
    assert 0.05 < np.median(y.tagwise_dispersion) < 0.15
    # the weights flow into the GLM fit
    fit = pe.glm_fit(y, design=design)
    assert np.all(np.isfinite(fit.coefficients))
