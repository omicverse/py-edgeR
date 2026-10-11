"""Vectorised ports of edgeR's compiled GLM kernels (edgeR 4.0.16).

Every function processes all genes at once with NumPy and follows the C/C++
source step by step (same start values, damping, tolerances and floors), so
results agree with Bioconductor edgeR to floating-point precision instead of
merely converging to the same optimum.

Sources: src/glm_levenberg.cpp, src/R_initialize_levenberg.cpp,
src/glm_one_group.cpp, src/nbdev.c, src/R_compute_apl.cpp, src/adj_coxreid.cpp,
R/mglmOneWay.R, R/designAsFactor.R, R/residDF.R, R/comboGroups.R.
"""
from __future__ import annotations

import numpy as np
from scipy import special

LOW_VALUE = 1e-10            # utils.h low_value
LOG_LOW_VALUE = np.log(LOW_VALUE)


# --------------------------------------------------------------------------- deviance
def unit_nb_deviance(y, mu, phi):
    """compute_nbdev.c compute_unit_nb_deviance (edgeR 4.10), elementwise with broadcasting."""
    y = np.asarray(y, dtype=float) + 1e-8
    mu = np.asarray(mu, dtype=float) + 1e-8
    y, mu, phi = np.broadcast_arrays(y, mu, np.asarray(phi, dtype=float))
    out = np.empty(y.shape)
    pois = phi < 1e-4
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = y[pois] - mu[pois]
        p = phi[pois]
        # nbdev.c writes 2/3*resid, which is C integer division (== 0); reproduced as such.
        out[pois] = 2 * (y[pois] * np.log(y[pois] / mu[pois]) - r - 0.5 * r * r * p * (1 + p * (0.0 * r - y[pois])))
        rest = ~pois
        prod = mu * phi
        gam = rest & (prod > 1e6)
        out[gam] = 2 * ((y[gam] - mu[gam]) / mu[gam] - np.log(y[gam] / mu[gam])) * mu[gam] / (1 + prod[gam])
        nb = rest & ~(prod > 1e6)
        inv = 1.0 / phi[nb]
        out[nb] = 2 * (y[nb] * np.log(y[nb] / mu[nb]) + (y[nb] + inv) * np.log((mu[nb] + inv) / (y[nb] + inv)))
    return np.maximum(out, 0.0)                 # edgeR >= 4.4 floors unit deviances at 0


def nb_deviance(Y, Mu, Phi, W):
    return (W * unit_nb_deviance(Y, Mu, Phi)).sum(axis=1)


# --------------------------------------------------------------------------- linear algebra helpers
def _xtwx(X, Wt):
    """X: (s, p); Wt: (n, s) -> (n, p, p)."""
    return np.einsum("ns,sp,sq->npq", Wt, X, X, optimize=True)


def _batched_cholesky(A):
    """Unpivoted Cholesky per matrix; returns (L, ok) like LAPACK dpotrf (fails on a pivot <= 0)."""
    n, p, _ = A.shape
    L = np.zeros_like(A)
    ok = np.ones(n, bool)
    for j in range(p):
        s = A[:, j, j] - np.einsum("nk,nk->n", L[:, j, :j], L[:, j, :j])
        good = ok & np.isfinite(s) & (s > 0)
        ok &= good
        d = np.sqrt(np.where(good, s, 1.0))
        L[:, j, j] = d
        if j + 1 < p:
            L[:, j + 1:, j] = (A[:, j + 1:, j] - np.einsum("nik,nk->ni", L[:, j + 1:, :j], L[:, j, :j])) / d[:, None]
    return L, ok


def _chol_solve(L, b):
    """Solve (L L^T) x = b by batched forward/back substitution."""
    n, p, _ = L.shape
    z = np.empty((n, p))
    for i in range(p):
        z[:, i] = (b[:, i] - np.einsum("nk,nk->n", L[:, i, :i], z[:, :i])) / L[:, i, i]
    x = np.empty((n, p))
    for i in range(p - 1, -1, -1):
        x[:, i] = (z[:, i] - np.einsum("nk,nk->n", L[:, i + 1:, i], x[:, i + 1:])) / L[:, i, i]
    return x


def ldl_logdet_floor(A):
    """sum(log D) of an LDL^T factorisation with edgeR's floor (adj_coxreid.cpp).

    For the positive (semi)definite XtWX matrices edgeR feeds to dsytrf the
    product of D equals det(A); exactly-zero pivots (all-zero columns of WX)
    are floored at log(1e-10), as in the C++ code.
    """
    n, p, _ = A.shape
    A = A.copy()
    out = np.zeros(n)
    for j in range(p):
        d = A[:, j, j].copy()
        bad = ~np.isfinite(d) | (d < LOW_VALUE)
        out += np.where(bad, LOG_LOW_VALUE, np.log(np.where(bad, 1.0, d)))
        if j + 1 < p:
            safe = np.where(np.abs(d) > 0, d, 1.0)
            col = A[:, j + 1:, j] / safe[:, None]
            col[np.abs(d) == 0] = 0.0
            A[:, j + 1:, j + 1:] -= col[:, :, None] * A[:, j, j + 1:][:, None, :]
    return out


# --------------------------------------------------------------------------- one-way layout
def design_as_factor(X):
    """R designAsFactor: integer codes 0..g-1 in order of sorted unique row keys."""
    X = np.asarray(X, dtype=float)
    z = (np.e + np.pi) / 5
    key = (X * z ** np.arange(X.shape[1])[None, :]).mean(axis=1)
    lev, codes = np.unique(key, return_inverse=True)       # R factor levels are sorted
    return codes, len(lev)


def mglm_one_group(Y, Off, Phi, W, start=None, maxit=50, tol=1e-10):
    """glm_one_group.cpp (vectorised): returns beta (n,), -inf for all-zero rows."""
    n, s = Y.shape
    nonzero = (Y > LOW_VALUE).any(axis=1)
    num = np.where(Y > LOW_VALUE, Y / np.exp(Off) * W, 0.0).sum(axis=1)
    with np.errstate(divide="ignore"):
        beta = np.log(num / W.sum(axis=1))
    if start is not None:                       # C code recomputes the start where it is NA
        start = np.asarray(start, dtype=float)
        beta = np.where(np.isnan(start), beta, start)
    act = nonzero.copy()
    for _ in range(maxit):
        i = np.flatnonzero(act)
        if not i.size:
            break
        mu = np.exp(beta[i, None] + Off[i])
        den = 1 + mu * Phi[i]
        step = ((Y[i] - mu) / den * W[i]).sum(1) / (mu / den * W[i]).sum(1)
        beta[i] += step
        act[i[np.abs(step) < tol]] = False
    beta[~nonzero] = -np.inf
    return beta


def mglm_one_way(Y, X, Off, Phi, W, start=None, maxit=50, tol=1e-10):
    codes, ng = design_as_factor(X)
    first = np.array([np.flatnonzero(codes == g)[0] for g in range(ng)])
    du = X[first]
    indicator = (du == 1).sum() == ng and (du == 0).sum() == (ng - 1) * ng
    gstart = None
    if start is not None:
        gstart = np.asarray(start, dtype=float) @ du.T
    beta = np.zeros((Y.shape[0], ng))
    for g in range(ng):
        j = codes == g
        beta[:, g] = mglm_one_group(Y[:, j], Off[:, j], Phi[:, j], W[:, j],
                                    start=None if gstart is None else gstart[:, g], maxit=maxit, tol=tol)
    beta = np.maximum(beta, -1e8)
    mu = np.exp(beta[:, codes] + Off)
    if not indicator:
        beta = np.linalg.solve(du, beta.T).T
    return beta, mu


def is_one_way(X):
    return design_as_factor(X)[1] == X.shape[1]


# --------------------------------------------------------------------------- Levenberg
def levenberg_start(Y, Off, Phi, W, X):
    """get_levenberg_start, 'null' method: least-squares fit of the constant log weighted mean."""
    N = np.exp(Off)
    cw = W * N / (1 + Phi * N)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log((Y * cw / N).sum(1) / cw.sum(1))
    Q, R = np.linalg.qr(X)
    coef_of_one = np.linalg.solve(R, Q.T @ np.ones(X.shape[0]))   # beta solving X b ~ 1
    return c[:, None] * coef_of_one[None, :]


def fit_levenberg(Y, Off, Phi, W, X, beta, tol=1e-6, maxit=200):
    """glm_levenberg.cpp, vectorised across genes. Returns beta, mu, deviance, iter, failed."""
    Y = np.asarray(Y, dtype=float)
    n, s = Y.shape
    p = X.shape[1]
    beta = np.array(beta, dtype=float, copy=True)
    ymax = Y.max(axis=1)
    zero = ymax < LOW_VALUE
    mu = np.zeros_like(Y)
    nz = ~zero
    mu[nz] = np.exp(Off[nz] + beta[nz] @ X.T)
    dev = np.zeros(n)
    dev[nz] = nb_deviance(Y[nz], mu[nz], Phi[nz], W[nz])
    beta[zero] = np.nan
    iters = np.zeros(n, int)
    failed = np.zeros(n, bool)
    max_info = np.full(n, -1.0)
    lam = np.zeros(n)
    act = nz.copy() & (p > 0)
    eye = np.eye(p)
    for it in range(1, maxit + 1):
        i = np.flatnonzero(act)
        if not i.size:
            break
        iters[i] = it
        m = mu[i]
        den = 1 + m * Phi[i]
        ww = m / den * W[i]
        deriv = (Y[i] - m) / den * W[i]
        A = _xtwx(X, ww)
        dl = deriv @ X
        max_info[i] = np.maximum(max_info[i], np.einsum("nii->ni", A).max(1))
        if it == 1:
            lam[i] = np.maximum(max_info[i] * 1e-6, 1e-13)
        # inner Levenberg loop, per gene
        pending = np.ones(i.size, bool)
        lev = np.zeros(i.size, int)
        low_dev = np.zeros(i.size, bool)
        dbeta = np.zeros((i.size, p))
        fail_i = np.zeros(i.size, bool)
        while pending.any():
            k = np.flatnonzero(pending)
            lev[k] += 1
            gi = i[k]
            # Cholesky with lambda*=10 on failure
            todo = np.ones(k.size, bool)
            L = np.zeros((k.size, p, p))
            while todo.any():
                t = np.flatnonzero(todo)
                Ad = A[k[t]] + lam[gi[t], None, None] * eye
                try:                                   # fast path: LAPACK, whole batch positive definite
                    L[t] = np.linalg.cholesky(Ad)
                    ok = np.ones(t.size, bool)
                except np.linalg.LinAlgError:
                    Lt, ok = _batched_cholesky(Ad)     # find the matrices dpotrf would reject
                    L[t[ok]] = Lt[ok]
                todo[t[ok]] = False
                bad = t[~ok]
                lam[gi[bad]] *= 10
                lam[gi[bad]] = np.where(lam[gi[bad]] <= 0, 1e-100, lam[gi[bad]])
            db = _chol_solve(L, dl[k])
            bn = beta[gi] + db
            mun = np.exp(Off[gi] + bn @ X.T)
            devn = nb_deviance(Y[gi], mun, Phi[gi], W[gi])
            ld = devn / ymax[gi] < 1e-13
            accept = (devn <= dev[gi]) | ld
            a = k[accept]
            ga = gi[accept]
            beta[ga], mu[ga], dev[ga] = bn[accept], mun[accept], devn[accept]
            dbeta[a] = db[accept]
            low_dev[a] = ld[accept]
            pending[a] = False
            r = k[~accept]
            gr = gi[~accept]
            dbeta[r] = db[~accept]
            lam[gr] *= 2
            lam[gr] = np.where(lam[gr] <= 0, 1e-100, lam[gr])
            f = lam[gr] / max_info[gr] > 1e13
            fail_i[r[f]] = True
            pending[r[f]] = False
        failed[i] |= fail_i
        divergence = np.einsum("np,np->n", dl, dbeta)
        stop = fail_i | low_dev | (divergence < tol)
        dec = (~stop) & (lev == 1)
        lam[i[dec]] /= 10
        act[i[stop]] = False
    return beta, mu, dev, iters, failed


# --------------------------------------------------------------------------- zero-fit helpers
def combo_groups(truths):
    """Rows grouped by identical TRUE/FALSE pattern (order of groups irrelevant downstream)."""
    truths = np.asarray(truths, bool)
    _, inv = np.unique(truths, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def qr_rank_columns(X, tol=1e-7):
    """Columns kept by R's qr() (LINPACK dqrdc2, limited pivoting): greedy in column order."""
    keep = []
    Q = np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        v = X[:, j].astype(float)
        nrm = np.linalg.norm(v)
        r = v - Q @ (Q.T @ v)
        if nrm > 0 and np.linalg.norm(r) >= tol * nrm:
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
            keep.append(j)
    return np.array(keep, int)


def resid_df(zero, X):
    """R .residDF."""
    zero = np.asarray(zero, bool)
    nlib, ncoef = zero.shape[1], X.shape[1]
    nzero = zero.sum(1)
    df = np.full(zero.shape[0], nlib - ncoef, dtype=float)
    df[nzero == nlib] = 0
    some = np.flatnonzero((nzero > 0) & (nzero < nlib))
    if some.size:
        for grp in combo_groups(zero[some]):
            rows = some[grp]
            z = zero[rows[0]]
            df[rows] = max(0, nlib - z.sum() - qr_rank_columns(X[~z]).size)
    return df


# --------------------------------------------------------------------------- adjusted profile likelihood
def compute_apl(Y, Mu, Phi, W, X, adjust=True):
    """R_compute_apl.cpp (vectorised)."""
    Y = np.asarray(Y, dtype=float)
    pos = Mu != 0
    w = W
    cmu = Mu * w
    cy = Y * w
    cd = Phi / w
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = 1.0 / cd
        logmur = np.log(cmu + rr)
        nb = (cy * np.log(cmu) - cy * logmur + rr * np.log(rr) - rr * logmur
              + special.gammaln(cy + rr) - special.gammaln(cy + 1) - special.gammaln(rr))
        po = cy * np.log(cmu) - cmu - special.gammaln(cy + 1)
    ll = np.where(cd > 0, nb, po) + np.log(w)
    out = np.where(pos, ll, 0.0).sum(1)
    if adjust:
        ww = np.where(pos, cmu / (1 + cd * cmu), 0.0)
        if X.shape[1] == 1:
            adj = np.log(np.abs(ww.sum(1))) / 2
        else:
            adj = ldl_logdet_floor(_xtwx(X, ww)) * 0.5
        out = out - adj
    return out


# --------------------------------------------------------------------------- maximizeInterpolant
def _fmm_spline(x, Y):
    """R stats splines.c fmm_spline for one x and many rows of y. Returns b, c, d (rows x n)."""
    n = x.size
    m = Y.shape[0]
    b = np.zeros((m, n)); c = np.zeros((m, n)); d = np.zeros(n)
    if n < 3:
        t = (Y[:, 1] - Y[:, 0]) / (x[1] - x[0])
        b[:, 0] = b[:, 1] = t
        return b, c, np.zeros((m, n))
    # 1-based indices from C mapped to 0-based
    d[0] = x[1] - x[0]
    c[:, 1] = (Y[:, 1] - Y[:, 0]) / d[0]
    bd = np.zeros(n)
    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        bd[i] = 2.0 * (d[i - 1] + d[i])
        c[:, i + 1] = (Y[:, i + 1] - Y[:, i]) / d[i]
        c[:, i] = c[:, i + 1] - c[:, i]
    bd[0] = -d[0]
    bd[n - 1] = -d[n - 2]
    c[:, 0] = 0.0
    c[:, n - 1] = 0.0
    if n > 3:
        c0 = c[:, 2] / (x[3] - x[1]) - c[:, 1] / (x[2] - x[0])
        cn = c[:, n - 2] / (x[n - 1] - x[n - 3]) - c[:, n - 3] / (x[n - 2] - x[n - 4])
        c[:, 0] = c0 * d[0] * d[0] / (x[3] - x[0])
        c[:, n - 1] = -cn * d[n - 2] * d[n - 2] / (x[n - 1] - x[n - 4])
    for i in range(1, n):
        t = d[i - 1] / bd[i - 1]
        bd[i] = bd[i] - t * d[i - 1]
        c[:, i] = c[:, i] - t * c[:, i - 1]
    c[:, n - 1] = c[:, n - 1] / bd[n - 1]
    for i in range(n - 2, -1, -1):
        c[:, i] = (c[:, i] - d[i] * c[:, i + 1]) / bd[i]
    b[:, n - 1] = (Y[:, n - 1] - Y[:, n - 2]) / d[n - 2] + d[n - 2] * (c[:, n - 2] + 2.0 * c[:, n - 1])
    D = np.zeros((m, n))
    for i in range(n - 1):
        b[:, i] = (Y[:, i + 1] - Y[:, i]) / d[i] - d[i] * (c[:, i + 1] + 2.0 * c[:, i])
        D[:, i] = (c[:, i + 1] - c[:, i]) / d[i]
        c[:, i] = 3.0 * c[:, i]
    c[:, n - 1] = 3.0 * c[:, n - 1]
    D[:, n - 1] = D[:, n - 2]
    return b, c, D


def maximize_interpolant(x, Y):
    """edgeR maximizeInterpolant (interpolator.cpp find_max), vectorised over rows."""
    x = np.asarray(x, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m, n = Y.shape
    rows = np.arange(m)
    k = np.argmax(Y, axis=1)                    # first maximum, as the C loop's strict '>'
    best = Y[rows, k].copy()
    xmax = x[k].copy()
    b, c, d = _fmm_spline(x, Y)

    def side(seg, base):
        a2, b2, c2 = 3 * d[rows, seg], 2 * c[rows, seg], b[rows, seg]
        disc = b2 * b2 - 4 * a2 * c2
        ok = disc >= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            sol = -b2 / (2 * a2) - np.sqrt(np.where(ok, disc, 0.0)) / (2 * a2)
        width = x[np.minimum(seg + 1, n - 1)] - x[seg]
        ok &= (sol > 0) & (sol < width)
        val = ((d[rows, seg] * sol + c[rows, seg]) * sol + b[rows, seg]) * sol + Y[rows, seg]
        return ok, sol + x[seg], val

    left = k > 0
    seg = np.where(left, k - 1, 0)
    ok, xs, val = side(seg, None)
    upd = left & ok & (val > best)
    best[upd], xmax[upd] = val[upd], xs[upd]
    right = k < n - 1
    seg = np.where(right, k, 0)
    ok, xs, val = side(seg, None)
    upd = right & ok & (val > best)
    best[upd], xmax[upd] = val[upd], xs[upd]
    return xmax
