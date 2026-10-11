"""limma 3.58 empirical-Bayes helpers used by edgeR: fitFDist / squeezeVar.

fitFDist with a covariate regresses on a natural cubic spline basis
(``splines::ns(covariate, df, intercept=TRUE)``). Fitted values and residual
effects depend only on the column space of that basis, which is reproduced
here exactly (B-splines on R's knots, constrained to zero second derivative at
the boundary knots, linear extrapolation outside them).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg, special
from scipy.interpolate import BSpline


def trigamma_inverse(x):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.full_like(x, np.nan)
    big, small = x > 1e7, (x < 1e-6) & (x >= 0)
    y[big] = 1 / np.sqrt(x[big])
    y[small] = 1 / x[small]
    mid = ~(big | small | (x < 0) | np.isnan(x))
    if mid.any():
        xm = x[mid]
        ym = 0.5 + 1 / xm
        for _ in range(51):
            tri = special.polygamma(1, ym)
            dif = tri * (1 - tri / xm) / special.polygamma(2, ym)
            ym = ym + dif
            if np.max(-dif / ym) < 1e-8:
                break
        y[mid] = ym
    return y


def _quantile7(x, p):
    return np.quantile(x, p, method="linear")


class NaturalSplineBasis:
    """Column space of splines::ns(x, df=df, intercept=True)."""

    def __init__(self, x, df):
        x = np.asarray(x, dtype=float)
        self.bk = np.array([x.min(), x.max()])
        nik = df - 2
        if nik > 0:
            knots = _quantile7(x, np.linspace(0, 1, nik + 2)[1:-1])
            lo, hi = self.bk
            if np.any(knots == lo):
                i = knots == lo
                knots[i] = knots[i] + (np.min(knots[knots > lo]) - lo) / 8
            if np.any(knots == hi):
                i = knots == hi
                knots[i] = knots[i] - (hi - np.max(knots[knots < hi])) / 8
        else:
            knots = np.zeros(0)
        self.t = np.sort(np.concatenate([np.repeat(self.bk, 4), knots]))
        self.nb = len(self.t) - 4
        const = np.vstack([self._design(self.bk, 2)])          # 2 x nb, second derivatives at boundaries
        self.N = linalg.null_space(const)                      # nb x (nb-2)

    def _design(self, x, deriv=0):
        out = np.empty((np.size(x), self.nb))
        for j in range(self.nb):
            c = np.zeros(self.nb); c[j] = 1.0
            b = BSpline(self.t, c, 3, extrapolate=False)
            if deriv:
                b = b.derivative(deriv)
            v = b(np.asarray(x, dtype=float))
            out[:, j] = np.nan_to_num(v)
        # right boundary: BSpline with extrapolate=False is open on the right end
        at_hi = np.asarray(x) == self.t[-1]
        if np.any(at_hi):
            for j in range(self.nb):
                c = np.zeros(self.nb); c[j] = 1.0
                b = BSpline(self.t, c, 3, extrapolate=True)
                if deriv:
                    b = b.derivative(deriv)
                out[at_hi, j] = b(np.asarray(x, dtype=float)[at_hi])
        return out

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        lo, hi = self.bk
        B = np.empty((x.size, self.nb))
        inside = (x >= lo) & (x <= hi)
        if inside.any():
            B[inside] = self._design(x[inside])
        for piv, mask in ((lo, x < lo), (hi, x > hi)):
            if mask.any():
                v0 = self._design(np.array([piv]))[0]
                v1 = self._design(np.array([piv]), 1)[0]
                B[mask] = v0[None, :] + (x[mask] - piv)[:, None] * v1[None, :]
        return B @ self.N


def fit_f_dist(x, df1, covariate=None):
    """limma::fitFDist (limma 3.58.1)."""
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return x.copy(), 0.0
    df1 = np.asarray(df1, dtype=float)
    scalar_df = df1.ndim == 0 or df1.size == 1
    if scalar_df:
        d = float(df1.reshape(-1)[0])
        if not (np.isfinite(d) and d > 1e-15):
            return np.nan, np.nan
        ok = np.ones(n, bool)
        df1 = np.full(n, d)
    else:
        ok = np.isfinite(df1) & (df1 > 1e-15)
    cov = None
    if covariate is not None:
        cov = np.asarray(covariate, dtype=float).copy()
        fin = np.isfinite(cov)
        if not fin.all():
            if fin.any():
                lo, hi = cov[fin].min(), cov[fin].max()
                cov[cov == -np.inf] = lo - 1
                cov[cov == np.inf] = hi + 1
            else:
                cov = np.sign(cov)
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 1:
        return x[ok][0], 0.0
    notallok = nok < n
    cov_bad = None
    if notallok:
        x = x[ok]
        df1 = df1[ok]
        if cov is not None:
            cov_bad = cov[~ok]
            cov = cov[ok]
    if cov is not None:
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, np.unique(cov).size)
        if splinedf < 2:
            s, d2 = fit_f_dist(x, df1)
            return np.full(n, s), d2
    x = np.maximum(x, 0)
    m = np.median(x)
    if m == 0:
        warnings.warn("More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) - special.digamma(df1 / 2) + np.log(df1 / 2)
    if cov is None:
        emean = np.mean(e)
        evar = np.sum((e - emean) ** 2) / (nok - 1)
    else:
        basis = NaturalSplineBasis(cov, splinedf)
        X = basis(cov)
        Q, R = np.linalg.qr(X)
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-7 * np.abs(R[0, 0])))
        coef, *_ = np.linalg.lstsq(X, e, rcond=None)
        fitted = X @ coef
        rss = np.sum((e - fitted) ** 2)
        evar = rss / (nok - rank)                    # mean(effects[-(1:rank)]^2)
        if notallok:
            emean = np.zeros(n)
            emean[ok] = fitted
            emean[~ok] = basis(cov_bad) @ coef
        else:
            emean = fitted
    evar = evar - np.mean(special.polygamma(1, df1 / 2))
    if evar > 0:
        df2 = 2 * trigamma_inverse(evar)[0]
        s20 = np.exp(emean + special.digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        s20 = np.mean(x) if cov is None else np.exp(emean)
    return s20, df2


def squeeze_var(var, df, covariate=None):
    """limma::squeezeVar, non-robust. Returns (df_prior, var_prior, var_post)."""
    var = np.asarray(var, dtype=float).copy()
    df = np.asarray(df, dtype=float)
    n = var.size
    if n == 0:
        raise ValueError("var is empty")
    if n == 1:
        return 0.0, var.copy(), var.copy()
    if df.size > 1:
        var[df == 0] = 0
    s20, df_prior = fit_f_dist(var, df, covariate=covariate)
    if np.any(np.isnan(np.atleast_1d(df_prior))):
        raise ValueError("Could not estimate prior df")
    vp = np.asarray(s20, dtype=float)
    if np.isfinite(df_prior):
        post = (df * var + df_prior * vp) / (df + df_prior)
    else:
        post = np.broadcast_to(vp, var.shape).astype(float).copy()
    return df_prior, s20, post
