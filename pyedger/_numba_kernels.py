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


# --------------------------------------------------------------------------- new QL method (ql_glm.c, ql_weights.c)
if HAVE_NUMBA:
    from ._ql_adjust import (NB_A_1_1, NB_A_1_2, NB_A_1_3, NB_A_2_1, NB_A_2_2, NB_K_1_1, NB_K_1_2, NB_K_1_3,
                             NB_K_2_1, NB_K_2_2, POIS_ALPHA_WEIGHTS, POIS_KAPPA_WEIGHTS)

    LOW_BOUND = 1e-32
    QL_THRESHOLD_ZERO = 1e-4
    QR_TOL = 1e-7

    @njit(cache=True)
    def _cheb1(w, it, x, m):
        # out = w[it] + w[it+1]*x + sum_i w[it+i]*T_i(x), as the C loops
        c0 = 1.0
        c1 = x
        out = w[it] + w[it + 1] * x
        for i in range(2, m):
            c2 = 2 * x * c1 - c0
            out += w[it + i] * c2
            c0 = c1
            c1 = c2
        return out

    @njit(cache=True)
    def _cheb_vec(x, m, out):
        out[0] = 1.0
        out[1] = x
        for i in range(2, m):
            out[i] = 2 * x * out[i - 1] - out[i - 2]

    @njit(cache=True)
    def _cheb2(w, it, xc, yc, m):
        out = 0.0
        for i in range(m):
            for j in range(m):
                out += w[it] * xc[j] * yc[i]
                it += 1
        return out

    @njit(cache=True)
    def _pois_alpha(mu):
        logmu = 0.0
        it = 0
        if mu < LOW_BOUND:
            return 0.0
        elif mu < 0.0200:
            x = 2 * mu / 0.02 - 1
            logmu = math.log(mu)
        elif mu < 0.4249:
            x = (2 * mu - 0.4449) / 0.4049; it = 10
        elif mu < 1.5000:
            x = (2 * mu - 1.9249) / 1.0751; it = 20
        elif mu < 3.5440:
            x = (2 * mu - 5.0440) / 2.0440; it = 30
        elif mu < 20.000:
            x = (2 * mu - 23.544) / 16.456; it = 40
        else:
            return 1 - 1 / (6 * mu) - 1 / (2 * mu * mu)
        out = _cheb1(POIS_ALPHA_WEIGHTS, it, x, 10)
        if mu < 0.0200:
            out = -out * logmu / ((1 + logmu) * (1 + logmu))
        return out

    @njit(cache=True)
    def _pois_kappa(mu):
        logmu = 0.0
        it = 0
        if mu < LOW_BOUND:
            return 0.0
        elif mu < 0.0200:
            x = 2 * mu / 0.02 - 1
            logmu = math.log(mu) / (1 + math.log(mu))
        elif mu < 0.4966:
            x = (2 * mu - 0.5166) / 0.4766; it = 10
        elif mu < 1.5000:
            x = (2 * mu - 1.9966) / 1.0034; it = 20
        elif mu < 4.2714:
            x = (2 * mu - 5.7714) / 2.7714; it = 30
        elif mu < 20.000:
            x = (2 * mu - 24.2714) / 15.7286; it = 40
        else:
            return 1 - 1 / (2.5 * mu * mu)
        out = _cheb1(POIS_KAPPA_WEIGHTS, it, x, 10)
        if mu < 0.0200:
            out = out * mu * logmu * logmu
        return out

    @njit(cache=True)
    def _mix(w, it, off2, off3, x, y, m):
        # w1, w2 in y (offsets 0, off2), w3 in x (offset off3); returns (w1, w2, w3)
        xc = np.empty(m)
        yc = np.empty(m)
        xc[0] = 1.0; xc[1] = x; yc[0] = 1.0; yc[1] = y
        w1 = w[it] + w[it + 1] * y
        w2 = w[it + off2] + w[it + off2 + 1] * y
        w3 = w[it + off3] + w[it + off3 + 1] * x
        for i in range(2, m):
            xc[i] = 2 * x * xc[i - 1] - xc[i - 2]
            yc[i] = 2 * y * yc[i - 1] - yc[i - 2]
            w1 += w[it + i] * yc[i]
            w2 += w[it + off2 + i] * yc[i]
            w3 += w[it + off3 + i] * xc[i]
        return w1, w2, w3

    @njit(cache=True)
    def _nbdevc_1(mu, phi, kappa):
        y = phi / 0.368 - 1
        if mu < LOW_BOUND:
            return 0.0
        tail = (1 - 1 / (2.5 * mu * mu)) if kappa else (1 - 1 / (6 * mu) - 1 / (2 * mu * mu))
        if mu > 60:
            it = 0
            if kappa and mu > 250:
                it = 18
            elif mu > 120:
                it = 12
            elif mu > 80:
                it = 6
            return _cheb1(NB_K_1_3 if kappa else NB_A_1_3, it, y, 6) * tail
        elif mu > 20:
            it = 0
            if mu < 25:
                x = (2 * mu - 45) / 5
            elif mu < 30:
                x = (2 * mu - 55) / 5; it = 7
            elif mu < 40:
                x = (mu - 35) / 5; it = 14
            else:
                x = (mu - 50) / 10; it = 21
            w1, w2, w3 = _mix(NB_K_1_2 if kappa else NB_A_1_2, it, 7, 35, x, y, 7)
            return (w2 + (w1 - w2) * w3) * tail
        else:
            it = 0
            if mu < 0.01:
                x = 200 * mu - 1
            elif mu < 0.33:
                x = (2 * mu - 0.34) / 0.32; it = 49
            elif kappa and mu < 1.30:
                x = (2 * mu - 1.63) / 0.97; it = 98
            elif (not kappa) and mu < 1.77:
                x = (2 * mu - 2.10) / 1.44; it = 98
            elif mu < 4.00:
                if kappa:
                    x = (2 * mu - 5.30) / 2.70
                else:
                    x = (2 * mu - 5.77) / 2.23
                it = 147
            elif mu < 10.0:
                x = (mu - 7) / 3; it = 196
            else:
                x = (mu - 15) / 5; it = 245
            xc = np.empty(7)
            yc = np.empty(7)
            _cheb_vec(x, 7, xc)
            _cheb_vec(y, 7, yc)
            out = _cheb2(NB_K_1_1 if kappa else NB_A_1_1, it, xc, yc, 7)
            return out * (_pois_kappa(mu) if kappa else _pois_alpha(mu))

    @njit(cache=True)
    def _nbdevc_2(mu, phi, kappa):
        y = phi / 2 - 1
        if mu < LOW_BOUND:
            return 0.0
        if mu < 50:
            it = 0
            if mu < 0.01:
                x = mu / 0.01 - 1
            elif kappa and mu < 0.50:
                x = (2 * mu - 0.51) / 0.49; it = 100
            elif (not kappa) and mu < 0.43:
                x = (mu - 0.22) / 0.21; it = 100
            elif kappa and mu < 3.88:
                x = (2 * mu - 4.38) / 3.38; it = 200
            elif (not kappa) and mu < 3.62:
                x = (2 * mu - 4.05) / 3.19; it = 200
            elif mu < 10.0:
                if kappa:
                    x = (2 * mu - 13.88) / 6.12
                else:
                    x = (2 * mu - 13.62) / 6.38
                it = 300
            elif mu < 30.0:
                x = mu / 10 - 2; it = 400
            else:
                x = mu / 10 - 4; it = 500
            xc = np.empty(10)
            yc = np.empty(10)
            _cheb_vec(x, 10, xc)
            _cheb_vec(y, 10, yc)
            out = _cheb2(NB_K_2_1 if kappa else NB_A_2_1, it, xc, yc, 10)
            if mu < 0.01:
                if kappa:
                    logmu = math.log(mu) / (1 + math.log(mu))
                    out = out * mu * logmu * logmu
                else:
                    logmu = math.log(mu)
                    out = out * (logmu / ((1 + logmu) * (1 + logmu)))
            return out
        elif mu < 5000:
            it = 0
            if mu < 100:
                x = 200 / mu - 3
            elif mu < 1000:
                x = (2000 / mu - 11) / 9; it = 10
            else:
                x = 2500 / mu - 1.5; it = 20
            w1, w2, w3 = _mix(NB_K_2_2 if kappa else NB_A_2_2, it, 10, 40, x, y, 10)
            return w1 + (w2 - w1) * w3
        else:
            return _cheb1(NB_K_2_2 if kappa else NB_A_2_2, 30, y, 10)

    @njit(cache=True)
    def _ql_weight(u, phi, prior):
        # compute_weight for phi < 4.001 (the exact branch is left to the NumPy path)
        mu = u / prior
        if phi < 0.736:
            return _nbdevc_1(mu, phi, False), _nbdevc_1(mu, phi, True)
        return _nbdevc_2(mu, phi, False), _nbdevc_2(mu, phi, True)

    @njit(cache=True)
    def _nrm2(x, r0, j):
        n = x.shape[0]
        s = 0.0
        for i in range(r0, n):
            s += x[i, j] * x[i, j]
        r = math.sqrt(s)
        if not (r > 1e-150 and r < 1e150):
            scale = 0.0
            for i in range(r0, n):
                a = abs(x[i, j])
                if a > scale:
                    scale = a
            if scale > 0:
                s = 0.0
                for i in range(r0, n):
                    t = abs(x[i, j]) / scale
                    s += t * t
                r = math.sqrt(s) * scale
            else:
                r = 0.0
        return r

    @njit(cache=True)
    def _qr_hat(x, hat):
        # src/ql_glm.c qr_hat: LINPACK dqrdc2 (tol 1e-7, limited pivoting) + dqrqy on rank columns
        n, p = x.shape
        qraux = np.empty(p)
        for j in range(p):
            qraux[j] = _nrm2(x, 0, j)
        work1 = qraux.copy()
        work2 = qraux.copy()
        for j in range(p):
            if work2[j] == 0.0:
                work2[j] = 1.0
        lup = min(n, p)
        k = p + 1
        col = np.empty(n)
        for l in range(lup):
            while not ((l + 1) >= k or qraux[l] >= work2[l] * QR_TOL):
                for i in range(n):
                    col[i] = x[i, l]
                for c in range(l, p - 1):
                    for i in range(n):
                        x[i, c] = x[i, c + 1]
                for i in range(n):
                    x[i, p - 1] = col[i]
                t, tt, ttt = qraux[l], work1[l], work2[l]
                for c in range(l, p - 1):
                    qraux[c] = qraux[c + 1]
                    work1[c] = work1[c + 1]
                    work2[c] = work2[c + 1]
                qraux[p - 1], work1[p - 1], work2[p - 1] = t, tt, ttt
                k -= 1
            if l + 1 == n:
                continue
            nrmxl = _nrm2(x, l, l)
            if nrmxl == 0.0:
                continue
            if x[l, l] != 0.0:
                nrmxl = math.copysign(nrmxl, x[l, l])
            sc = 1.0 / nrmxl
            for i in range(l, n):
                x[i, l] *= sc
            x[l, l] = 1.0 + x[l, l]
            for j in range(l + 1, p):
                d = 0.0
                for i in range(l, n):
                    d += x[i, l] * x[i, j]
                t = -d / x[l, l]
                for i in range(l, n):
                    x[i, j] += t * x[i, l]
                if qraux[j] == 0.0:
                    continue
                tt = abs(x[l, j]) / qraux[j]
                tt = 1.0 - tt * tt
                if tt < 0.0:
                    tt = 0.0
                if abs(tt) < 1e-6:
                    qraux[j] = _nrm2(x, l + 1, j)
                    work1[j] = qraux[j]
                else:
                    qraux[j] = qraux[j] * math.sqrt(tt)
            qraux[l] = x[l, l]
            x[l, l] = -nrmxl
        k = min(k - 1, n)
        Y = np.zeros((n, k))
        for i in range(k):
            Y[i, i] = 1.0
        ju = min(k, n - 1)
        for j in range(ju - 1, -1, -1):
            if qraux[j] == 0.0:
                continue
            v0 = qraux[j]
            for c in range(j, k):
                d = v0 * Y[j, c]
                for i in range(j + 1, n):
                    d += x[i, j] * Y[i, c]
                t = -d / v0
                Y[j, c] += v0 * t
                for i in range(j + 1, n):
                    Y[i, c] += x[i, j] * t
        for i in range(n):
            h = 0.0
            for c in range(k):
                h += Y[i, c] * Y[i, c]
            hat[i] = h

    @njit(parallel=True, cache=True)
    def ql_adjust_nb(Y, Mu, X, Disp, W, prior, want_mat):
        # compute_adjust_vec / compute_adjust_mat (src/ql_glm.c), per gene
        G, n = Y.shape
        p = X.shape[1]
        df = np.zeros(G)
        dev = np.zeros(G)
        s2 = np.zeros(G)
        gm = G if want_mat else 1
        udm = np.zeros((gm, n))
        ufm = np.zeros((gm, n))
        lvm = np.zeros((gm, n))
        for g in prange(G):
            xw = np.empty((n, p))
            hat = np.zeros(n)
            for i in range(n):
                u = Mu[g, i]
                zw = math.sqrt(u / (1 + (u * Disp[g, i] * W[g, i] / prior)))
                for c in range(p):
                    xw[i, c] = X[i, c] * zw
            _qr_hat(xw, hat)
            dv = 0.0
            dfg = 0.0
            for i in range(n):
                wa, wk = _ql_weight(Mu[g, i], Disp[g, i], prior / W[g, i])
                udp = _unit_dev(Y[g, i], Mu[g, i], Disp[g, i] * W[g, i] / prior)
                hdp = 1.0 - hat[i]
                if hdp < QL_THRESHOLD_ZERO:
                    udp = 0.0
                    hdp = 0.0
                dv += (udp * wa) * W[g, i]
                dfg += hdp * wk
                if want_mat:
                    udm[g, i] = udp * wa
                    ufm[g, i] = hdp * wk
                    lvm[g, i] = hat[i]
            dev[g] = dv
            df[g] = dfg
            s2[g] = 0.0 if dfg < QL_THRESHOLD_ZERO else dv / dfg
        return df, dev, s2, udm, ufm, lvm

    @njit(cache=True)
    def rsort_with_index_nb(x):
        # R's rsort_with_index (Shell sort); returns the permutation
        n = x.size
        xs = x.copy()
        idx = np.arange(n)
        h = 1
        while h <= n // 9:
            h = 3 * h + 1
        while h > 0:
            for i in range(h, n):
                v = xs[i]
                iv = idx[i]
                j = i
                while j >= h and xs[j - h] > v:
                    xs[j] = xs[j - h]
                    idx[j] = idx[j - h]
                    j -= h
                xs[j] = v
                idx[j] = iv
            h //= 3
        return idx

    @njit(cache=True)
    def _lowest(x, y, n, xs, nleft, nright, w, it, rw):
        # edgeR src/clowess.c lowest (0-based arrays, 1-based nleft/nright); returns (ys, notok)
        rng = x[n - 1] - x[0]
        h = max(xs - x[nleft - 1], x[nright - 1] - xs)
        h9 = 0.999 * h
        h1 = 0.001 * h
        a = 0.0
        j = nleft
        while j <= n:
            w[j - 1] = 0.0
            r = abs(x[j - 1] - xs)
            if r <= h9:
                if r <= h1:
                    w[j - 1] = 1.0
                else:
                    q = r / h
                    q = 1.0 - q * q * q
                    w[j - 1] = q * q * q
                if it > 1:
                    w[j - 1] *= rw[j - 1]
                a += w[j - 1]
            elif x[j - 1] > xs:
                break
            j += 1
        nrt = j - 1
        if a <= 0.0:
            return 0.0, True
        for j in range(nleft, nrt + 1):
            w[j - 1] /= a
        if h > 0.0:
            a = 0.0
            for j in range(nleft, nrt + 1):
                a += w[j - 1] * x[j - 1]
            b = xs - a
            c = 0.0
            for j in range(nleft, nrt + 1):
                d = x[j - 1] - a
                c += w[j - 1] * (d * d)
            if math.sqrt(c) > 0.001 * rng:
                b /= c
                for j in range(nleft, nrt + 1):
                    w[j - 1] *= (b * (x[j - 1] - a) + 1.0)
        ys = 0.0
        for j in range(nleft, nrt + 1):
            ys += w[j - 1] * y[j - 1]
        return ys, False

    @njit(cache=True)
    def clowess_nb(x, y, f, nsteps, delta):
        # edgeR src/clowess.c clowess (x sorted)
        n = x.size
        ys = np.zeros(n)
        if n < 2:
            ys[0] = y[0]
            return ys
        rw = np.zeros(n)
        res = np.zeros(n)
        ns = max(2, min(n, int(f * n + 1e-7)))
        it = 1
        while it <= nsteps + 1:
            nleft = 1
            nright = ns
            last = 0
            i = 1
            while True:
                if nright < n:
                    d1 = x[i - 1] - x[nleft - 1]
                    d2 = x[nright] - x[i - 1]
                    if d1 > d2:
                        nleft += 1
                        nright += 1
                        continue
                v, notok = _lowest(x, y, n, x[i - 1], nleft, nright, res, it, rw)
                ys[i - 1] = y[i - 1] if notok else v
                if last < i - 1:
                    denom = x[i - 1] - x[last - 1]
                    for j in range(last + 1, i):
                        alpha = (x[j - 1] - x[last - 1]) / denom
                        ys[j - 1] = alpha * ys[i - 1] + (1.0 - alpha) * ys[last - 1]
                last = i
                cut = x[last - 1] + delta
                i = last + 1
                while i <= n:
                    if x[i - 1] > cut:
                        break
                    if x[i - 1] == x[last - 1]:
                        ys[i - 1] = ys[last - 1]
                        last = i
                    i += 1
                i = max(last + 1, i - 1)
                if last >= n:
                    break
            for k in range(n):
                res[k] = y[k] - ys[k]
            sc = 0.0
            for k in range(n):
                sc += abs(res[k])
            sc /= n
            if it > nsteps:
                break
            for k in range(n):
                rw[k] = abs(res[k])
            srt = np.sort(rw)
            m1 = n // 2
            if n % 2 == 0:
                m2 = n - m1 - 1
                cmad = 3.0 * (srt[m1] + srt[m2])
            else:
                cmad = 6.0 * srt[m1]
            if cmad < 1e-7 * sc:
                break
            c9 = 0.999 * cmad
            c1 = 0.001 * cmad
            for k in range(n):
                r = abs(res[k])
                if r <= c1:
                    rw[k] = 1.0
                elif r <= c9:
                    q = r / cmad
                    q = 1.0 - q * q
                    rw[k] = q * q
                else:
                    rw[k] = 0.0
            it += 1
        return ys

    @njit(parallel=True, cache=True)
    def one_group_nb(Y, Off, Phi, W, start, has_start, maxit, tol):
        # glm.c glm_one_group_vec per gene (non-converged rows keep the last iterate)
        n, s = Y.shape
        beta = np.empty(n)
        for g in prange(n):
            allzero = True
            num = 0.0
            tot = 0.0
            for j in range(s):
                if Y[g, j] > LOW_VALUE:
                    num += Y[g, j] / math.exp(Off[g, j]) * W[g, j]
                    allzero = False
                tot += W[g, j]
            b = math.log(num / tot) if num > 0 else -math.inf
            if has_start and not math.isnan(start[g]):
                b = start[g]
            if allzero:
                beta[g] = -math.inf
                continue
            for _ in range(maxit):
                dl = 0.0
                info = 0.0
                for j in range(s):
                    mu = math.exp(b + Off[g, j])
                    den = 1 + mu * Phi[g, j]
                    dl += (Y[g, j] - mu) / den * W[g, j]
                    info += mu / den * W[g, j]
                step = dl / info
                b += step
                if abs(step) < tol:
                    break
            beta[g] = b
        return beta
