"""Compiled per-gene kernels (numba), parallel over genes.

These follow edgeR's C code gene by gene, as the vectorised NumPy versions in
``_glm_batch`` do, but without (genes x samples) temporaries. They are used
when numba is installed; set ``PYEDGER_NO_NUMBA=1`` to force the NumPy path.
The number of threads follows numba's ``NUMBA_NUM_THREADS``.
"""
from __future__ import annotations

import math
import os

import numpy as np

try:
    if os.environ.get("PYEDGER_NO_NUMBA"):
        raise ImportError
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                   # pragma: no cover
    HAVE_NUMBA = False

LOW_VALUE = 1e-10
LOG_LOW_VALUE = math.log(LOW_VALUE)

if HAVE_NUMBA:

    @njit(cache=True, fastmath=False)
    def _unit_dev(y, mu, phi):
        # compute_nbdev.c compute_unit_nb_deviance (edgeR 4.10)
        y = y + 1e-8
        mu = mu + 1e-8
        if phi < 1e-4:
            r = y - mu
            # C writes 2/3*resid: integer division, == 0
            out = 2 * (y * math.log(y / mu) - r - 0.5 * r * r * phi * (1 + phi * (0.0 * r - y)))
        else:
            prod = mu * phi
            if prod > 1e6:
                out = 2 * ((y - mu) / mu - math.log(y / mu)) * mu / (1 + prod)
            else:
                inv = 1.0 / phi
                out = 2 * (y * math.log(y / mu) + (y + inv) * math.log((mu + inv) / (y + inv)))
        return out if out > 0.0 else 0.0

    @njit(cache=True)
    def _gene_dev(y, mu, phi, w):
        d = 0.0
        for j in range(y.size):
            d += w[j] * _unit_dev(y[j], mu[j], phi[j])
        return d

    @njit(cache=True)
    def _chol(A, L):
        # unpivoted Cholesky; False where LAPACK dpotrf would fail (pivot <= 0 / NaN)
        p = A.shape[0]
        for j in range(p):
            s = A[j, j]
            for k in range(j):
                s -= L[j, k] * L[j, k]
            if not (s > 0.0):
                return False
            d = math.sqrt(s)
            L[j, j] = d
            for i in range(j + 1, p):
                t = A[i, j]
                for k in range(j):
                    t -= L[i, k] * L[j, k]
                L[i, j] = t / d
            for i in range(j):
                L[i, j] = 0.0
        return True

    @njit(cache=True)
    def _chol_solve1(L, b, x):
        p = b.size
        z = np.empty(p)
        for i in range(p):
            t = b[i]
            for k in range(i):
                t -= L[i, k] * z[k]
            z[i] = t / L[i, i]
        for i in range(p - 1, -1, -1):
            t = z[i]
            for k in range(i + 1, p):
                t -= L[k, i] * x[k]
            x[i] = t / L[i, i]

    @njit(cache=True)
    def _fit_mu(off, X, beta, mu):
        s, p = X.shape
        for j in range(s):
            eta = off[j]
            for k in range(p):
                eta += X[j, k] * beta[k]
            mu[j] = math.exp(eta)

    @njit(parallel=True, cache=True)
    def fit_levenberg_nb(Y, Off, Phi, W, X, beta0, tol, maxit):
        n, s = Y.shape
        p = X.shape[1]
        beta = beta0.copy()
        MU = np.zeros((n, s))
        dev = np.zeros(n)
        iters = np.zeros(n, np.int64)
        failed = np.zeros(n, np.bool_)
        for g in prange(n):
            y = Y[g]; off = Off[g]; phi = Phi[g]; w = W[g]
            ymax = 0.0
            for j in range(s):
                if y[j] > ymax:
                    ymax = y[j]
            if ymax < LOW_VALUE:
                for k in range(p):
                    beta[g, k] = np.nan
                continue
            b = beta[g].copy()
            mu = np.empty(s)
            _fit_mu(off, X, b, mu)
            d = _gene_dev(y, mu, phi, w)
            if p == 0:
                MU[g] = mu; dev[g] = d
                continue
            A = np.empty((p, p)); Ad = np.empty((p, p)); L = np.zeros((p, p))
            dl = np.empty(p); db = np.empty(p); bn = np.empty(p); mun = np.empty(s)
            ww = np.empty(s); der = np.empty(s)
            max_info = -1.0
            lam = 0.0
            fail = False
            for it in range(1, maxit + 1):
                iters[g] = it
                for j in range(s):
                    den = 1 + mu[j] * phi[j]
                    ww[j] = mu[j] / den * w[j]
                    der[j] = (y[j] - mu[j]) / den * w[j]
                for a in range(p):
                    t = 0.0
                    for j in range(s):
                        t += der[j] * X[j, a]
                    dl[a] = t
                    for c in range(a + 1):
                        t = 0.0
                        for j in range(s):
                            t += ww[j] * X[j, a] * X[j, c]
                        A[a, c] = t
                        A[c, a] = t
                for a in range(p):
                    if A[a, a] > max_info:
                        max_info = A[a, a]
                if it == 1:
                    lam = max_info * 1e-6
                    if lam < 1e-13:
                        lam = 1e-13
                lev = 0
                low_dev = False
                dbeta_dot = 0.0
                while True:
                    lev += 1
                    while True:
                        for a in range(p):
                            for c in range(p):
                                Ad[a, c] = A[a, c]
                            Ad[a, a] += lam
                        if _chol(Ad, L):
                            break
                        lam *= 10
                        if lam <= 0:
                            lam = 1e-100
                    _chol_solve1(L, dl, db)
                    for k in range(p):
                        bn[k] = b[k] + db[k]
                    _fit_mu(off, X, bn, mun)
                    dn = _gene_dev(y, mun, phi, w)
                    ld = dn / ymax < 1e-13
                    dbeta_dot = 0.0
                    for k in range(p):
                        dbeta_dot += dl[k] * db[k]
                    if dn <= d or ld:
                        for k in range(p):
                            b[k] = bn[k]
                        for j in range(s):
                            mu[j] = mun[j]
                        d = dn
                        low_dev = ld
                        break
                    lam *= 2
                    if lam <= 0:
                        lam = 1e-100
                    if lam / max_info > 1e13:
                        fail = True
                        break
                if fail or low_dev or dbeta_dot < tol:
                    break
                if lev == 1:
                    lam /= 10
            failed[g] = fail
            beta[g] = b
            MU[g] = mu
            dev[g] = d
        return beta, MU, dev, iters, failed

    @njit(parallel=True, cache=True)
    def compute_apl_nb(Y, Mu, Phi, W, X, adjust):
        # R_compute_apl / compute_apl.c with edgeR's LDL logdet floor
        n, s = Y.shape
        p = X.shape[1]
        out = np.zeros(n)
        for g in prange(n):
            ll = 0.0
            ww = np.zeros(s)
            for j in range(s):
                m = Mu[g, j]
                if m == 0:
                    continue
                w = W[g, j]
                cmu = m * w
                cy = Y[g, j] * w
                cd = Phi[g, j] / w
                if cd > 0:
                    rr = 1.0 / cd
                    lmr = math.log(cmu + rr)
                    v = (cy * math.log(cmu) - cy * lmr + rr * math.log(rr) - rr * lmr
                         + math.lgamma(cy + rr) - math.lgamma(cy + 1) - math.lgamma(rr))
                else:
                    v = cy * math.log(cmu) - cmu - math.lgamma(cy + 1)
                ll += v + math.log(w)
                ww[j] = cmu / (1 + cd * cmu)
            if adjust:
                if p == 1:
                    t = 0.0
                    for j in range(s):
                        t += ww[j]
                    ll -= math.log(abs(t)) / 2
                else:
                    A = np.empty((p, p))
                    for a in range(p):
                        for c in range(a + 1):
                            t = 0.0
                            for j in range(s):
                                t += ww[j] * X[j, a] * X[j, c]
                            A[a, c] = t
                            A[c, a] = t
                    logdet = 0.0
                    for j in range(p):
                        dj = A[j, j]
                        if not (dj >= LOW_VALUE) or math.isinf(dj):
                            logdet += LOG_LOW_VALUE
                        else:
                            logdet += math.log(dj)
                        if j + 1 < p:
                            for i in range(j + 1, p):
                                col = A[i, j] / dj if dj != 0 else 0.0
                                for k in range(j + 1, p):
                                    A[i, k] -= col * A[j, k]
                    ll -= logdet * 0.5
            out[g] = ll
        return out
