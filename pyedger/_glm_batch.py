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

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import special

LOW_VALUE = 1e-10            # utils.h low_value
LOG_LOW_VALUE = np.log(LOW_VALUE)


# --------------------------------------------------------------------------- threading
def n_threads() -> int:
    """Threads for row-block parallelism: PYEDGER_NUM_THREADS, else numba's setting, else 1."""
    env = os.environ.get("PYEDGER_NUM_THREADS")
    if env:
        return max(1, int(env))
    from . import _numba_kernels as nk
    if nk.HAVE_NUMBA:
        import numba
        return max(1, int(numba.get_num_threads()))
    return 1


def block_map(fn, n, min_rows=1000):
    """Call ``fn(slice)`` on contiguous row blocks on a thread pool and concatenate the results."""
    nt = n_threads()
    if nt == 1 or n < 2 * min_rows:
        return fn(slice(0, n))
    nblk = min(nt * 4, n // min_rows)
    edges = np.linspace(0, n, nblk + 1).astype(int)
    with ThreadPoolExecutor(nt) as ex:
        res = list(ex.map(fn, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]))
    if isinstance(res[0], tuple):
        return tuple(np.concatenate(parts, axis=0) for parts in zip(*res))
    return np.concatenate(res, axis=0)


def row_parallel(fn, n, *args, min_rows=1000):
    """Apply ``fn`` to contiguous row blocks of the (n, ...) array arguments on a thread pool.

    NumPy/SciPy ufuncs release the GIL, so vectorised code on separate blocks runs
    on separate cores. Arguments whose first dimension is not ``n`` are passed whole.
    Outputs (an array or a tuple of arrays) are concatenated along the rows.
    """
    nt = n_threads()
    if nt == 1 or n < 2 * min_rows:
        return fn(*args)
    nblk = min(nt * 4, n // min_rows)
    edges = np.linspace(0, n, nblk + 1).astype(int)
    blocks = [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
    pick = lambda a, b: a[b] if isinstance(a, np.ndarray) and a.ndim and a.shape[0] == n else a
    with ThreadPoolExecutor(nt) as ex:
        res = list(ex.map(lambda b: fn(*[pick(a, b) for a in args]), blocks))
    if isinstance(res[0], tuple):
        return tuple(np.concatenate(parts, axis=0) for parts in zip(*res))
    return np.concatenate(res, axis=0)


# --------------------------------------------------------------------------- deviance
def unit_nb_deviance(y, mu, phi):
    """compute_nbdev.c compute_unit_nb_deviance (edgeR 4.10), elementwise with broadcasting."""
    y = np.asarray(y, dtype=float) + 1e-8
    mu = np.asarray(mu, dtype=float) + 1e-8
    y, mu, phi = np.broadcast_arrays(y, mu, np.asarray(phi, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv = 1.0 / phi
        out = 2 * (y * np.log(y / mu) + (y + inv) * np.log((mu + inv) / (y + inv)))     # NB
        pois = phi < 1e-4
        if pois.any():
            yp, mp, pp = y[pois], mu[pois], phi[pois]
            r = yp - mp
            # nbdev.c writes 2/3*resid, which is C integer division (== 0); reproduced as such.
            out[pois] = 2 * (yp * np.log(yp / mp) - r - 0.5 * r * r * pp * (1 + pp * (0.0 * r - yp)))
        gam = ~pois & (mu * phi > 1e6)
        if gam.any():
            yg, mg = y[gam], mu[gam]
            out[gam] = 2 * ((yg - mg) / mg - np.log(yg / mg)) * mg / (1 + mg * phi[gam])
    return np.maximum(out, 0.0, out=out)        # edgeR >= 4.4 floors unit deviances at 0


def nb_deviance(Y, Mu, Phi, W):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        return (W * unit_nb_deviance(Y, Mu, Phi)).sum(axis=1)
    sh = Y.shape
    return row_parallel(lambda y, m, p, w: (w * unit_nb_deviance(y, m, p)).sum(axis=1), sh[0],
                        Y, _full(Mu, sh), _full(Phi, sh), _full(W, sh))


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
    from . import _numba_kernels as nk
    n, s = Y.shape
    if nk.HAVE_NUMBA and n:
        Y = np.ascontiguousarray(Y, dtype=float)
        st = np.zeros(n) if start is None else np.ascontiguousarray(np.broadcast_to(np.asarray(start, dtype=float), (n,)))
        return nk.one_group_nb(Y, _full(Off, Y.shape), _full(Phi, Y.shape), _full(W, Y.shape), st,
                               start is not None, int(maxit), float(tol))
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


def _mglm_one_group_np(Y, Off, Phi, W, start=None, maxit=50, tol=1e-10):
    """glm_one_group.cpp vectorised over genes; converged genes leave the active set.

    mu = exp(beta) * exp(offset): exp(offset) is computed once instead of
    exp(beta + offset) per observation and iteration (equal up to rounding).
    """
    N = np.exp(Off)
    nonzero = (Y > LOW_VALUE).any(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(np.where(Y > LOW_VALUE, Y / N * W, 0.0).sum(axis=1) / W.sum(axis=1))
    if start is not None:                       # C code recomputes the start where it is NA
        start = np.asarray(start, dtype=float)
        beta = np.where(np.isnan(start), beta, start)
    act = np.flatnonzero(nonzero)
    y, n_, p, w = Y[act], N[act], Phi[act], W[act]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(maxit):
            if not act.size:
                break
            mu = np.exp(beta[act])[:, None] * n_
            den = 1 + mu * p
            step = ((y - mu) / den * w).sum(axis=1) / (mu / den * w).sum(axis=1)
            beta[act] += step
            moving = ~(np.abs(step) < tol)
            if not moving.all():
                act = act[moving]
                y, n_, p, w = y[moving], n_[moving], p[moving], w[moving]
    beta[~nonzero] = -np.inf
    return beta


def _one_way_layout(X):
    codes, ng = design_as_factor(X)
    first = np.array([np.flatnonzero(codes == g)[0] for g in range(ng)])
    du = X[first]
    indicator = (du == 1).sum() == ng and (du == 0).sum() == (ng - 1) * ng
    return codes, ng, du, indicator


def _fit_one_way_block(Y, Off, Phi, W, gstart, codes, ng, maxit, tol, deviance=True):
    beta = np.column_stack([
        _mglm_one_group_np(Y[:, codes == g], Off[:, codes == g], Phi[:, codes == g], W[:, codes == g],
                           start=None if gstart is None else gstart[:, g], maxit=maxit, tol=tol)
        for g in range(ng)])
    beta = np.maximum(beta, -1e8)
    mu = np.exp(beta[:, codes] + Off)
    dev = (W * unit_nb_deviance(Y, mu, Phi)).sum(axis=1) if deviance else np.full(len(Y), np.nan)
    return beta, mu, dev


def fit_one_way(Y, X, Off, Phi, W, start=None, maxit=50, tol=1e-10, deviance=True):
    """mglmOneWay followed by the NB deviance: returns beta, mu, deviance (row blocks in parallel).

    ``deviance=False`` skips the deviance (NaN), for callers that only need the fit.
    """
    Y = np.asarray(Y, dtype=float)
    sh = Y.shape
    phi_row = _row_constant(Phi, sh)
    if phi_row is not None and _all_ones(W):
        # one dispersion per gene and unit weights (glmFit, glmQLFit, the dispersion grid):
        # broadcast the dispersion instead of materialising (genes x samples) matrices
        beta, mu, _ = OneWayFitter(Y, X, Off).fit(phi_row, start=start, maxit=maxit, tol=tol)
        if not deviance:
            return beta, mu, np.full(sh[0], np.nan)
        dev = block_map(lambda sl: unit_nb_deviance(Y[sl], mu[sl], phi_row[sl, None]).sum(axis=1), sh[0])
        return beta, mu, dev
    codes, ng, du, indicator = _one_way_layout(X)
    gstart = None if start is None else np.asarray(start, dtype=float) @ du.T
    beta, mu, dev = row_parallel(
        lambda y, o, p, w, gs: _fit_one_way_block(y, o, p, w, gs, codes, ng, maxit, tol, deviance),
        sh[0], Y, _full(Off, sh), _full(Phi, sh), _full(W, sh), gstart)
    if not indicator:
        beta = np.linalg.solve(du, beta.T).T
    return beta, mu, dev


class OneWayFitter:
    """mglmOneWay prepared once for repeated fits of the same counts at different dispersions.

    The dispersion grid of estimateDisp refits the same genes 21 times with a single
    dispersion value per fit. Group sub-matrices and exp(offset) are built once here;
    each fit then only broadcasts the dispersion. Unit weights are assumed.
    """

    def __init__(self, Y, X, Off):
        codes, ng, du, indicator = _one_way_layout(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        Off = np.broadcast_to(np.asarray(Off, dtype=float), Y.shape)
        self.codes, self.ng, self.du, self.indicator = codes, ng, du, indicator
        self.Off = Off
        self.cols = [np.flatnonzero(codes == g) for g in range(ng)]
        self.Yg = [np.ascontiguousarray(Y[:, c]) for c in self.cols]
        self.Ng = [np.exp(Off[:, c]) for c in self.cols]
        self.n = Y.shape[0]

    def fit(self, phi, start=None, maxit=50, tol=1e-10):
        """phi: scalar or one value per gene. Returns beta (group coefficients), mu, log(mu)."""
        phi = np.asarray(phi, dtype=float)
        phi_col = phi.reshape(-1, 1) if phi.ndim else phi
        gstart = None if start is None else np.asarray(start, dtype=float) @ self.du.T

        def block(sl):
            pc = phi_col[sl] if phi_col.ndim else phi_col
            beta = np.column_stack([
                _one_group_newton(self.Yg[g][sl], self.Ng[g][sl], pc, None,
                                  None if gstart is None else gstart[sl, g], maxit, tol)
                for g in range(self.ng)])
            beta = np.maximum(beta, -1e8)
            log_mu = beta[:, self.codes] + self.Off[sl]
            return beta, np.exp(log_mu), log_mu

        beta, mu, log_mu = block_map(block, self.n)
        if not self.indicator:
            beta = np.linalg.solve(self.du, beta.T).T
        return beta, mu, log_mu


def _one_group_newton(Y, N, phi, W, start, maxit, tol):
    """glm_one_group.cpp vectorised over genes: Y, N = exp(offset) (n, s); phi (n, 1), (n, s) or scalar; W None = unit."""
    nonzero = (Y > LOW_VALUE).any(axis=1)
    wsum = Y.shape[1] if W is None else W.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(Y > LOW_VALUE, Y / N, 0.0) if W is None else np.where(Y > LOW_VALUE, Y / N * W, 0.0)
        beta = np.log(q.sum(axis=1) / wsum)
    if start is not None:                       # C code recomputes the start where it is NA
        beta = np.where(np.isnan(start), beta, start)
    act = np.flatnonzero(nonzero)
    y, n_ = Y[act], N[act]
    p = phi[act] if np.ndim(phi) == 2 else phi
    w = None if W is None else W[act]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(maxit):
            if not act.size:
                break
            mu = np.exp(beta[act])[:, None] * n_
            r = mu * p                                   # r = w / (1 + mu * phi), built in place
            r += 1.0
            np.reciprocal(r, out=r)
            if w is not None:
                r *= w
            info = np.einsum("ij,ij->i", mu, r)          # sum(mu * r) without a temporary
            step = (np.einsum("ij,ij->i", y, r) - info) / info
            beta[act] += step
            moving = ~(np.abs(step) < tol)
            if not moving.all():
                act = act[moving]
                y, n_ = y[moving], n_[moving]
                if np.ndim(p) == 2:
                    p = p[moving]
                if w is not None:
                    w = w[moving]
    beta[~nonzero] = -np.inf
    return beta


def compute_apl_one_disp(Y, mu, log_mu, phi, X, lgamma_y1):
    """R_compute_apl.cpp for unit weights and one dispersion per gene (scalar or (n,)).

    log(mu) is taken from the fit, and the dispersion-only terms r log r - lgamma(r)
    are evaluated once per gene instead of once per observation.
    """
    phi = np.asarray(phi, dtype=float)
    X = np.asarray(X, dtype=float)

    def block(sl):
        y, m, lm, lg = Y[sl], mu[sl], log_mu[sl], lgamma_y1[sl]
        ph = phi[sl].reshape(-1, 1) if phi.ndim else phi
        if not np.all(ph > 0):
            return _compute_apl_np(y, m, np.broadcast_to(ph, y.shape), np.ones_like(y), X, True, lg)
        rr = 1.0 / ph
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = y * lm - (y + rr) * np.log(m + rr) + special.gammaln(y + rr) - lg
            const = rr * np.log(rr) - special.gammaln(rr)
        pos = m != 0
        if pos.all():
            out = ll.sum(1) + (const * y.shape[1] if np.ndim(const) == 0 else const[:, 0] * y.shape[1])
        else:
            out = np.where(pos, ll + const, 0.0).sum(1)
        ww = m / (1 + ph * m)
        if not pos.all():
            ww = np.where(pos, ww, 0.0)
        if X.shape[1] == 1:
            adj = np.log(np.abs(ww.sum(1))) / 2
        else:
            adj = ldl_logdet_floor(_xtwx(X, ww)) * 0.5
        return out - adj

    return block_map(block, Y.shape[0])


def mglm_one_way(Y, X, Off, Phi, W, start=None, maxit=50, tol=1e-10):
    beta, mu, _ = fit_one_way(Y, X, Off, Phi, W, start=start, maxit=maxit, tol=tol)
    return beta, mu


def _row_constant(A, shape):
    """The per-row value if ``A`` (broadcastable to ``shape``) is constant along rows, else None."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 0:
        return np.full(shape[0], float(A))
    if A.ndim == 1:
        return A if A.shape[0] == shape[0] else None
    if A.shape[1] == 1:
        return A[:, 0]
    return A[:, 0].copy() if (A == A[:, :1]).all() else None


def _all_ones(W):
    W = np.asarray(W)
    return bool((W == 1).all())


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


def _full(a, shape):
    return np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=float), shape))


def fit_levenberg(Y, Off, Phi, W, X, beta, tol=1e-6, maxit=200):
    """glm_levenberg.cpp per gene. Returns beta, mu, deviance, iter, failed."""
    from . import _numba_kernels as nk
    if nk.HAVE_NUMBA:
        Y = np.ascontiguousarray(Y, dtype=float)
        X = np.ascontiguousarray(X, dtype=float)
        return nk.fit_levenberg_nb(Y, _full(Off, Y.shape), _full(Phi, Y.shape), _full(W, Y.shape), X,
                                   _full(beta, (Y.shape[0], X.shape[1])), float(tol), int(maxit))
    return _fit_levenberg_np(Y, Off, Phi, W, X, beta, tol=tol, maxit=maxit)


def _fit_levenberg_np(Y, Off, Phi, W, X, beta, tol=1e-6, maxit=200):
    """glm_levenberg.cpp, vectorised across genes (NumPy fallback)."""
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
def compute_apl(Y, Mu, Phi, W, X, adjust=True, lgamma_y1=None):
    """R_compute_apl.cpp, vectorised; row blocks run in parallel.

    ``lgamma_y1`` = lgamma(Y * W + 1) does not depend on the dispersion and may be
    passed in to reuse it across the dispersion grid.
    """
    Y = np.asarray(Y, dtype=float)
    sh = Y.shape
    X = np.asarray(X, dtype=float)
    return row_parallel(lambda y, m, p, w, lg: _compute_apl_np(y, m, p, w, X, adjust, lg),
                        sh[0], Y, _full(Mu, sh), _full(Phi, sh), _full(W, sh),
                        None if lgamma_y1 is None else np.asarray(lgamma_y1, dtype=float))


def _compute_apl_np(Y, Mu, Phi, W, X, adjust=True, lgamma_y1=None):
    """R_compute_apl.cpp (vectorised)."""
    pos = Mu != 0
    w = W
    cmu = Mu * w
    cy = Y * w
    cd = Phi / w
    lgy1 = special.gammaln(cy + 1) if lgamma_y1 is None else lgamma_y1
    with np.errstate(divide="ignore", invalid="ignore"):
        if cd.size and (cd == cd[:, :1]).all():          # one dispersion per gene: per-row terms
            rr_row = 1.0 / cd[:, :1]
            rr = np.broadcast_to(rr_row, cd.shape)
            rr_log_rr = rr_row * np.log(rr_row)
            lg_rr = special.gammaln(rr_row)
        else:
            rr = 1.0 / cd
            rr_log_rr = rr * np.log(rr)
            lg_rr = special.gammaln(rr)
        log_cmu = np.log(cmu)
        logmur = np.log(cmu + rr)
        nb = (cy * log_cmu - cy * logmur + rr_log_rr - rr * logmur + special.gammaln(cy + rr) - lgy1 - lg_rr)
        if (cd > 0).all():
            ll = nb
        else:
            ll = np.where(cd > 0, nb, cy * log_cmu - cmu - lgy1)
    if not (w == 1).all():
        ll = ll + np.log(w)
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
