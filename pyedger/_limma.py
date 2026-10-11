"""limma 3.68 empirical-Bayes helpers used by edgeR: fitFDist / squeezeVar.

fitFDist with a covariate regresses on a natural cubic spline basis
(``splines::ns(covariate, df, intercept=TRUE)``). Fitted values and residual
effects depend only on the column space of that basis, which is reproduced
here exactly (B-splines on R's knots, constrained to zero second derivative at
the boundary knots, linear extrapolation outside them).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg, special, stats
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


def logmdigamma(x):
    """statmod::logmdigamma: log(x) - digamma(x) without cancellation (x > 0)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 5
    if np.any(small):
        xs = x[small]
        out[small] = (np.log(xs / (xs + 5)) + logmdigamma(xs + 5)
                      + 1 / xs + 1 / (xs + 1) + 1 / (xs + 2) + 1 / (xs + 3) + 1 / (xs + 4))
    big = ~small
    if np.any(big):
        z = x[big]
        t = 1 / z ** 2
        tail = t * (-1 / 12 + t * (1 / 120 + t * (-1 / 252 + t * (1 / 240 + t * (-1 / 132 + t * (
            691 / 32760 + t * (-1 / 12 + 3617 * t / 8160)))))))
        out[big] = 1 / (2 * z) - tail
    return out


def choose_lowess_span(n=1000, small_n=50, min_span=0.3, power=1 / 3):
    """limma::chooseLowessSpan."""
    return np.minimum(min_span + (1 - min_span) * (small_n / np.asarray(n, dtype=float)) ** power, 1.0)


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
    e = np.log(x) + logmdigamma(df1 / 2)
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
        s20 = np.exp(emean - logmdigamma(np.asarray(df2 / 2)))
    else:
        df2 = np.inf
        s20 = np.mean(x) if cov is None else np.exp(emean)
    return s20, df2


def _r_order(x):
    return np.argsort(x, kind="stable")


def _bh(p):
    n = p.size
    o = np.argsort(-p, kind="stable")
    q = np.minimum.accumulate(p[o] * n / np.arange(n, 0, -1))
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def _weighted_lowess_delta(x_sorted, npts):
    if npts >= x_sorted.size:
        return 0.0
    dx = np.sort(np.diff(x_sorted))
    cumrange = np.cumsum(dx)
    num = np.arange(npts)
    return float(np.min(cumrange[dx.size - num - 1] / (npts - num)))


def weighted_lowess(x, y, weights, span=0.3, iterations=4, npts=200):
    """limma::weightedLowess (C weighted_lowess) with output.style="loess"; returns fitted."""
    thr = 1e-7
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); w = np.asarray(weights, dtype=float)
    o = _r_order(x)
    xs, ys, ws = x[o], y[o], w[o]
    n = xs.size
    delta = _weighted_lowess_delta(xs, int(npts + 0.5))
    # find_seeds
    seeds = [0]
    last = 0
    for pt in range(1, n - 1):
        if xs[pt] - xs[last] > delta:
            seeds.append(pt)
            last = pt
    seeds.append(n - 1)
    seeds = np.asarray(seeds)
    total = 0.0
    for v in ws:                    # sequential sum as in C
        total += v
    spanweight = total * span
    subrange = (xs[-1] - xs[0]) / n
    # find_limits: points join the frame nearest-first (ties go right), until
    # the accumulated prior weight reaches spanweight
    starts, ends, dists = [], [], []
    for c in seeds:
        nl, nr = c, n - 1 - c
        dl = xs[c] - xs[c - np.arange(1, nl + 1)]
        dr = xs[c + np.arange(1, nr + 1)] - xs[c]
        dist = np.concatenate([dr, dl])
        side = np.concatenate([np.zeros(nr, int), np.ones(nl, int)])
        step = np.concatenate([np.arange(1, nr + 1), np.arange(1, nl + 1)])
        order = np.lexsort((side, dist))          # by distance, right before left on ties
        ww = np.where(side[order] == 0, ws[np.minimum(c + step[order], n - 1)], ws[np.maximum(c - step[order], 0)])
        cum = ws[c] + np.cumsum(ww)
        if ws[c] >= spanweight or order.size == 0:
            k = 0
        else:
            hit = np.flatnonzero(cum >= spanweight)
            k = (hit[0] + 1) if hit.size else order.size
        taken = order[:k]
        left = c - (step[taken][side[taken] == 1].max() if np.any(side[taken] == 1) else 0)
        right = c + (step[taken][side[taken] == 0].max() if np.any(side[taken] == 0) else 0)
        mdist = dist[taken].max() if k else 0.0
        while left > 0 and xs[left] == xs[left - 1]:
            left -= 1
        while right < n - 1 and xs[right] == xs[right + 1]:
            right += 1
        starts.append(left); ends.append(right); dists.append(mdist)
    rob = np.ones(n)
    fit = np.empty(n)
    for _ in range(int(iterations)):
        last = 0
        for j, c in enumerate(seeds):
            l, r, dmax = starts[j], ends[j], dists[j]
            xx, yy, wr = xs[l:r + 1], ys[l:r + 1], ws[l:r + 1] * rob[l:r + 1]
            if dmax < thr:
                fit[c] = np.sum(yy * wr) / np.sum(wr)
            else:
                wk = (1 - (np.abs(xs[c] - xx) / dmax) ** 3) ** 3 * wr
                aw = np.sum(wk)
                xm = np.sum(wk * xx) / aw
                ym = np.sum(wk * yy) / aw
                var = np.sum((xx - xm) ** 2 * wk)
                if var < thr:
                    fit[c] = ym
                else:
                    slope = np.sum((xx - xm) * (yy - ym) * wk) / var
                    fit[c] = slope * xs[c] + (ym - slope * xm)
            if j and c - last > 1:
                cur = xs[c] - xs[last]
                sub = np.arange(last + 1, c)
                if cur > thr * subrange:
                    slope = (fit[c] - fit[last]) / cur
                    fit[sub] = slope * xs[sub] + (fit[c] - slope * xs[c])
                else:
                    fit[sub] = 0.5 * (fit[c] + fit[last])
            last = c
        if _ + 1 == int(iterations):
            break
        res = np.abs(ys - fit)
        scale = np.mean(res)
        ro = np.argsort(res, kind="stable")
        rs = res[ro]
        cw = np.cumsum(ws[ro])
        half = total / 2
        k = int(np.searchsorted(cw, half))
        cmad = 3 * (rs[k] + rs[k + 1]) if cw[k] == half else 6 * rs[k]
        if cmad <= thr * scale:
            break
        rob = np.where(res < cmad, (1 - (res / cmad) ** 2) ** 2, 0.0)
    out = np.empty(n)
    out[o] = fit
    return out


def loess_fit(y, x, weights=None, span=0.3, iterations=4, min_weight=1e-5, max_weight=1e5):
    """limma::loessFit (method="weightedLowess"); returns fitted values."""
    from ._ql_adjust import r_lowess
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    n = y.size
    fitted = np.full(n, np.nan)
    obs = np.isfinite(y) & np.isfinite(x)
    xo, yo = x[obs], y[obs]
    nobs = yo.size
    if nobs == 0:
        return fitted
    if span < 1 / nobs:
        fitted[obs] = yo
        return fitted
    min_weight = max(min_weight, 0)
    if weights is not None:
        wo = np.asarray(weights, dtype=float)[obs]
        wo = np.where(np.isnan(wo), 0, wo)
        wo = np.minimum(np.maximum(wo, min_weight), max_weight)
        if wo.max() - wo.min() < 1e-15:
            weights = None
    if weights is None:
        o = _r_order(xo)
        _, lo = r_lowess(xo, yo, f=span, iter=iterations - 1)
        f = np.empty(nobs); f[o] = lo
        fitted[obs] = f
        return fitted
    nw = nobs if min_weight > 0 else int(np.sum(wo > 0))
    if nw < 4 + 1 / span:
        if nw == 1:
            fitted[obs] = yo[wo > 0]
        else:
            X = np.column_stack([np.ones(nobs), xo]); sw = np.sqrt(wo)
            coef, *_ = np.linalg.lstsq(X * sw[:, None], yo * sw, rcond=None)
            fitted[obs] = X @ coef
        return fitted
    fitted[obs] = weighted_lowess(xo, yo, wo, span=span, iterations=iterations, npts=200)
    return fitted


def fit_f_dist_unequal_df1(x, df1, covariate=None, span=None, robust=False, prior_weights=None):
    """limma::fitFDistUnequalDF1 (limma 3.68). Returns dict(scale, df2[, df2_shrunk])."""
    from .core import _brent_fmin
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    df1 = np.broadcast_to(np.asarray(df1, dtype=float), (n,)).copy() if np.ndim(df1) == 0 or np.size(df1) == 1 \
        else np.asarray(df1, dtype=float).copy()
    if prior_weights is not None:
        prior_weights = np.asarray(prior_weights, dtype=float).copy()
    if np.any(np.isnan(x)):
        i = np.isnan(x)
        if prior_weights is None:
            prior_weights = (~i).astype(float)
        else:
            prior_weights[i] = 0
        x[i] = 0
    if np.min(df1) < 0.01:
        i = df1 < 0.01
        if prior_weights is None:
            prior_weights = (~i).astype(float)
        else:
            prior_weights[i] = 0
        df1[i] = 1
    pw = prior_weights is not None
    inf = x > 0
    if pw:
        inf[prior_weights == 0] = False
    ninf = int(np.sum(inf))
    if ninf < 2:
        return dict(scale=np.nan, df2=np.nan)
    if ninf == 2:
        covariate, robust, prior_weights, pw = None, False, None, False
    m = np.median(x[inf])
    xpos = np.maximum(x, 1e-12 * m)
    z = np.log(xpos)
    d1 = df1 / 2
    e = z + logmdigamma(d1)
    w = 1 / special.polygamma(1, d1)
    if pw:
        w = w * prior_weights
    if covariate is None:
        emean = np.sum(w * e) / np.sum(w)
    else:
        if span is None:
            span = float(choose_lowess_span(n, small_n=500))
        emean = loess_fit(e, np.asarray(covariate, dtype=float), weights=w / np.quantile(w, 0.75, method="linear"),
                          min_weight=1e-8, max_weight=1e2, span=span, iterations=1)
    d1x = d1 * xpos
    pwt = prior_weights if pw else 1.0

    def m2ll(par):
        d2 = par / (1 - par)
        d2s20 = d2 * np.exp(emean - logmdigamma(np.asarray(d2)))
        return -2 * np.sum(pwt * (-(d1 + d2) * np.log1p(d1x / d2s20) - d1 * np.log(d2s20)
                                  + special.gammaln(d1 + d2) - special.gammaln(d2)))

    par = _brent_fmin(m2ll, 0.5, 0.9998, np.finfo(float).eps ** 0.25)
    d2 = par / (1 - par)
    s20 = np.exp(emean - logmdigamma(np.asarray(d2)))
    if not robust:
        return dict(scale=s20, df2=2 * d2)
    df2 = 2 * d2
    fstat = x / s20
    right = stats.f.sf(fstat, df1, df2)
    left = 1 - right
    if np.min(left) < 0.001:
        i = left < 0.001
        left[i] = stats.f.cdf(fstat[i], df1[i], df2)
    fdr = _bh(2 * np.minimum(left, right))
    fdr[fdr > 0.3] = 1
    if np.min(fdr) == 1:
        return dict(scale=s20, df2=df2)
    outpw = fit_f_dist_unequal_df1(x, df1, covariate=covariate, robust=False, prior_weights=fdr)
    s20, df2 = outpw["scale"], outpw["df2"]
    r = stats.rankdata(fstat)
    uniform = (n - r + 0.5) / n
    pnot = np.minimum(right / uniform, 1)
    if np.min(pnot) == 1:
        return outpw
    i = int(np.argmin(right))
    if right[i] == 0:
        df2_outlier = 0.0
        shrunk = pnot * df2
    else:
        df2_outlier = np.log(0.5) / np.log(right[i]) * df2
        new = stats.f.logsf(fstat[i], df1[i], df2_outlier)
        df2_outlier = np.log(0.5) / new * df2_outlier
        shrunk = pnot * df2 + (1 - pnot) * df2_outlier
    o = np.argsort(right, kind="stable")
    ordered = shrunk[o]
    mm = np.cumsum(ordered) / np.arange(1, n + 1)
    imin = int(np.argmin(mm))
    ordered[: imin + 1] = mm[imin]
    shrunk[o] = np.maximum.accumulate(ordered)
    return dict(scale=s20, df2=df2, df2_outlier=df2_outlier, df2_shrunk=shrunk)


def _squeeze_post(var, df, var_prior, df_prior):
    """limma .squeezeVar."""
    dfp = np.asarray(df_prior, dtype=float)
    if np.isfinite(np.max(dfp)):
        return (df * var + dfp * var_prior) / (df + dfp)
    n = var.size
    post = np.broadcast_to(np.asarray(var_prior, dtype=float), (n,)).astype(float).copy()
    if np.min(dfp) > 1e100:
        return post
    i = np.flatnonzero(np.isfinite(np.broadcast_to(dfp, (n,))))
    d = df[i] if np.size(df) > 1 else df
    dp = np.broadcast_to(dfp, (n,))[i]
    post[i] = (d * var[i] + dp * post[i]) / (d + dp)
    return post


def squeeze_var(var, df, covariate=None, span=None, robust=False, legacy=None):
    """limma::squeezeVar (limma 3.68). Returns (df_prior, var_prior, var_post).

    ``legacy=None`` uses fitFDist when the positive residual df are all equal
    and fitFDistUnequalDF1 otherwise. Robust estimation is implemented for the
    unequal-df method only.
    """
    var = np.asarray(var, dtype=float).copy()
    df = np.asarray(df, dtype=float)
    n = var.size
    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return 0.0, var.copy(), var.copy()
    if df.size > 1:
        var[df == 0] = 0
    if span is not None:
        legacy = False
    if legacy is None:
        dfp = df[df > 0]
        legacy = bool(np.min(dfp) == np.max(dfp))
    if legacy:
        if robust:
            raise NotImplementedError("robust=True with equal residual df (fitFDistRobustly)")
        s20, df_prior = fit_f_dist(var, df, covariate=covariate)
    else:
        fit = fit_f_dist_unequal_df1(var, df, covariate=covariate, span=span, robust=robust)
        s20 = fit["scale"]
        df_prior = fit.get("df2_shrunk", fit["df2"])
    if np.any(np.isnan(np.atleast_1d(df_prior))):
        raise ValueError("Could not estimate prior df")
    return df_prior, s20, _squeeze_post(var, df, s20, df_prior)
