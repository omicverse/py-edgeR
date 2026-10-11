"""NumPy port of edgeR's ``locfitByCol`` (and the parts of the R ``locfit``
package it relies on).

edgeR (R/locfitByCol.R) does, for every column ``j`` of ``y``::

    y[, j] <- fitted(locfit(y[, j] ~ x, weights = weights, alpha = span, deg = degree))

This module reproduces exactly what locfit 1.5-9.x does for that call
(1-D, ``family = "qgaussian"``, identity link, ``kern = "tricube"``,
``scale = FALSE``, default ``ev = rbox()`` i.e. the adaptive tree with
``cut = 0.8``, ``maxk = 100``).  The relevant C sources are
``startlf.c``, ``lf_nbhd.c``, ``locfit.c``, ``m_max.c``, ``m_jacob.c``,
``m_eigen.c``, ``pcomp.c``, ``procv.c``, ``ev_atree.c``, ``ev_main.c``,
``ev_interp.c`` and ``fitted.c``.  Summary of the algorithm:

* Bandwidth at a fit point ``v``: ``k = (int)(n * alpha + 1e-12)``;
  ``h`` is the k-th smallest ``|x_i - v|`` (``k < n``), the largest
  distance times ``k/n`` when ``k >= n``, and 0 when ``k == 0``.
* Kernel weights: tricube ``(1 - u^3)^3`` with ``u = |x_i - v| / h``
  (``h == 0``: weight 1 for points exactly at ``v``); multiplied by the
  prior weights.
* Local fit: locally constant (deg 0) or locally linear (deg 1) weighted
  least squares on the basis ``[1, x_i - v]``.  locfit computes it as the
  weighted mean start value plus a single Newton step solved through the
  diagonally-scaled Jacobi eigen-decomposition (``JAC_EIGD``), including its
  treatment of eigenvalues ``<= 1e-8 * max`` (those components are *not*
  divided).  This is reproduced literally.
* Parametric component: because the family is (quasi-)Gaussian, locfit
  first fits a *global* polynomial of the same degree (centred at the
  prior-weighted mean of ``x``), subtracts it from the vertex coefficients
  and adds it back at interpolation time.
* Evaluation structure: vertices at ``min(x)`` and ``max(x)``; a cell
  ``[a, b]`` is split at its midpoint while ``0.8 < (b - a) / hmin`` where
  ``hmin`` is the smaller positive bandwidth of its two vertices
  (``hmin == 0``: score ``2 (b - a) / (max x - min x)``).
* ``fitted()`` descends the tree for every data point (``2 (x - a) < b - a``
  goes left) and interpolates within the terminal cell: linear interpolation
  of vertex values for deg 0 (no derivative is stored for local constant
  fits), cubic Hermite interpolation of value and slope for deg 1.

Because the Gaussian fit is linear in the response and the evaluation
structure depends only on ``x``, the tree, bandwidths and kernel weights are
computed once and shared by all columns.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["locfit_by_col"]

_CUT = 0.8          # rbox() default cut
_MAXK = 100         # locfit.raw default maxk
_E_TOL = 1.0e-8     # m_eigen.c E_TOL


# --------------------------------------------------------------------------
# small linear algebra: max_nr single Newton step with JAC_EIGD (p = 1 or 2)
# --------------------------------------------------------------------------
def _eigd_solve(Z, f1):
    """Replicates jacob_solve(J, f1) with J->st == JAC_RAW -> JAC_EIGD.

    Z : (p, p) float array (raw X'WX); f1 : (p, ncol) array.
    Returns (delta (p, ncol), rank).
    """
    p = Z.shape[0]
    Z = Z.copy()
    dg = np.array([0.0 if Z[i, i] <= 0 else 1.0 / math.sqrt(Z[i, i]) for i in range(p)])
    for i in range(p):
        for j in range(p):
            Z[i, j] *= dg[i] * dg[j]
    # eig_dec (Jacobi), P starts as identity
    P = np.eye(p)
    for _it in range(20):
        ms = 0
        for i in range(p):
            for j in range(i + 1, p):
                if Z[i, j] * Z[i, j] > 1.0e-15 * abs(Z[i, i] * Z[j, j]):
                    c = (Z[j, j] - Z[i, i]) / 2
                    s = -Z[i, j]
                    r = math.sqrt(c * c + s * s)
                    c /= r
                    s = math.sqrt((1 - c) / 2) * (2 * (s > 0) - 1)
                    c = math.sqrt((1 + c) / 2)
                    for k in range(p):
                        u = Z[i, k]; v = Z[j, k]
                        Z[i, k] = u * c + v * s
                        Z[j, k] = v * c - u * s
                    for k in range(p):
                        u = Z[k, i]; v = Z[k, j]
                        Z[k, i] = u * c + v * s
                        Z[k, j] = v * c - u * s
                    Z[i, j] = Z[j, i] = 0.0
                    for k in range(p):
                        u = P[k, i]; v = P[k, j]
                        P[k, i] = u * c + v * s
                        P[k, j] = v * c - u * s
                    ms = 1
        if ms == 0:
            break
    # jacob_solve / eig_solve
    x = f1 * dg[:, None]
    mx = Z[0, 0]
    for i in range(1, p):
        if Z[i, i] > mx:
            mx = Z[i, i]
    tol = _E_TOL * mx
    rank = 0
    w = []
    for i in range(p):
        wi = 0.0
        for j in range(p):
            wi = wi + P[j, i] * x[j]
        if Z[i, i] > tol:
            wi = wi / Z[i, i]
            rank += 1
        w.append(wi)
    out = np.empty_like(x)
    for i in range(p):
        xi = 0.0
        for j in range(p):
            xi = xi + P[i, j] * w[j]
        out[i] = xi
    out *= dg[:, None]
    return out, rank


def _seqsum(a):
    """Left-to-right sum over axis 0 (like a C accumulation loop)."""
    a = np.asarray(a)
    if a.shape[0] == 0:
        return np.zeros(a.shape[1:]) if a.ndim > 1 else 0.0
    return np.cumsum(a, axis=0)[-1]


def _local_fit(dx, kw, pw, y, deg, prev_cf):
    """locfit() for one fit point given the in-window data.

    dx : x_i - v (window), kw : kernel weights (>0), pw : prior weights,
    y : (m, ncol) responses.  Returns cf (p, ncol), status ('ok'|'nopt').
    """
    # All sums are sequential (cumsum) in the order locfit visits the
    # neighbourhood, with the same association of products as the C code,
    # so that rounding matches R closely.
    ncol = y.shape[1]
    a = kw * pw                                   # ww * prwt (= ww*ZDDLL)
    s0 = _seqsum(a)
    if s0 == 0:
        # reginit returns LF_NOPT before touching des->cf: stale coefficients
        return prev_cf.copy(), "nopt"
    s1 = _seqsum(kw[:, None] * (pw[:, None] * y))  # w * ZDLL(LINIT) = w*(prwt*y)
    cf0 = (s1 - 0.0) / s0
    zd = pw[:, None] * (y - cf0[None, :])         # ZDLL = prwt*(y - th)
    p = 1 if deg == 0 else 2
    if p == 1:
        Z = np.array([[_seqsum(a * 1.0 * 1.0)]])
        f1 = _seqsum((1.0 * kw)[:, None] * zd)[None, :]
    else:
        Z00 = _seqsum(a * 1.0 * 1.0)
        Z01 = _seqsum(a * 1.0 * dx)
        Z10 = _seqsum(a * dx * 1.0)
        Z11 = _seqsum(a * dx * dx)
        Z = np.array([[Z00, Z01], [Z10, Z11]])
        f1 = np.vstack([_seqsum((1.0 * kw)[:, None] * zd),
                        _seqsum((dx * kw)[:, None] * zd)])
    delta, rank = _eigd_solve(Z, f1)
    if rank == 0:
        # max_nr: delta[0] = -f/f1[0]; f = sum ww * (-pw*(y-th)^2/2)
        z = y - cf0[None, :]
        f = _seqsum(kw[:, None] * (-pw[:, None] * z * z / 2))
        delta = delta.copy()
        delta[0] = -f / delta[0]
    cf = np.zeros((p, ncol))
    cf[0] = cf0
    cf = cf + 1.0 * delta
    return cf, "ok"


# --------------------------------------------------------------------------
# one-dimensional locfit with a shared design for many responses
# --------------------------------------------------------------------------
class _Locfit1D:
    def __init__(self, x, w, alpha, deg):
        self.x = np.asarray(x, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.n = n = self.x.size
        self.alpha = float(alpha)
        self.deg = int(deg)
        self.nn = int(n * self.alpha + 1e-12)
        order = np.argsort(self.x, kind="stable")
        self.order = order
        self.xs = self.x[order]
        self.ws = self.w[order]
        # startlf(): lfd.ord (x non-decreasing) -> nbhd() uses nbhd1()
        is_ord = bool(np.all(self.x[1:] >= self.x[:-1]))
        self.use_nbhd1 = is_ord and self.nn <= n
        # atree_guessnv -> maximum number of vertices
        a0 = 1.0 if self.alpha > 1 else 1.0 / self.alpha
        cu = min(1.0, _CUT)
        vc = 2
        nvm = int((5 * a0 / cu + 1) * vc)
        self.nvm = int((_MAXK / 100.0) * nvm)

    # ---- bandwidth (nbhd / nbhd1 / compbandwid)
    def bandwidth(self, v):
        n, k, xs = self.n, self.nn, self.xs
        if k == 0:
            return 0.0
        if k < n:
            i0 = int(np.searchsorted(xs, v))
            lo = max(0, i0 - k)
            hi = min(n, i0 + k)
            d = np.abs(xs[lo:hi] - v)
            return float(np.partition(d, k - 1)[k - 1])
        dmax = max(abs(xs[0] - v), abs(xs[-1] - v))
        return dmax * math.exp(math.log(1.0 * k / n) / 1)

    # ---- neighbourhood: kernel weights > 0, in the order locfit visits them
    def window(self, v, h):
        """Return (positions into the sorted arrays, kernel weights), ordered
        as locfit's nbhd1() (sorted x) or nbhd() (unsorted x) would."""
        xs = self.xs
        if h > 0:
            pad = 4 * h * 1e-12 + 1e-300
            lo = int(np.searchsorted(xs, v - h - pad, side="left"))
            hi = int(np.searchsorted(xs, v + h + pad, side="right"))
            di = np.abs(xs[lo:hi] - v) / 1.0
            u = di / h
            t = 1.0 - u * u * u
            kw = np.where(u > 1, 0.0, t * t * t)
        else:
            lo = int(np.searchsorted(xs, v, side="left"))
            hi = int(np.searchsorted(xs, v, side="right"))
            kw = np.ones(hi - lo)
        pos = kw > 0
        if self.use_nbhd1:
            # nbhd1(): closest point z, look left from z then right from z+1,
            # each direction stopping at the first zero weight.
            n = self.n
            if v <= xs[0]:
                z = 0
            elif v >= xs[n - 1]:
                z = n - 1
            else:
                l, r = 0, n - 1
                while r - l > 1:
                    z = (r + l) // 2
                    if xs[z] > v:
                        r = z
                    else:
                        l = z
                z = r if (v - xs[l]) > (xs[r] - v) else l
            if xs[z] > v:
                z -= 1
            zl = z - lo  # local index
            # (positions outside [lo, hi) have zero weight)
            left = np.empty(0, dtype=np.int64)
            if 0 <= zl < pos.size:
                zeros = np.flatnonzero(~pos[: zl + 1])
                start = zeros[-1] + 1 if zeros.size else 0
                left = np.arange(zl, start - 1, -1)
            j0 = zl + 1
            right = np.empty(0, dtype=np.int64)
            if 0 <= j0 < pos.size:
                seg = pos[j0:]
                zeros = np.flatnonzero(~seg)
                stop = j0 + (zeros[0] if zeros.size else seg.size)
                right = np.arange(j0, stop)
            loc = np.concatenate([left, right]).astype(np.int64)
        else:
            loc = np.flatnonzero(pos)
            loc = loc[np.argsort(self.order[lo + loc], kind="stable")]
        return lo + loc, kw[loc]

    def fit(self, Y):
        """Y : (n, ncol) in original order. Returns fitted (n, ncol)."""
        n, deg = self.n, self.deg
        ncol = Y.shape[1]
        Ys = Y[self.order]
        xs, ws = self.xs, self.ws
        p = 1 if deg == 0 else 2

        # ---- parametric component (compparcomp): global fit at xbar
        x0, w0 = self.x, self.w
        sw = _seqsum(w0)
        xbar = _seqsum(x0 * w0) / sw
        pc_cf, st = _local_fit(x0 - xbar, np.ones(n), w0, Y, deg, np.zeros((p, ncol)))
        if st != "ok":
            raise ValueError("compparcomp: no points in dataset?")
        self._last_cf = pc_cf

        # ---- vertices
        xv, hv, pseudo, vals = [], [], [], []
        lo_v, hi_v = [], []

        def fit_vertex(v):
            h = self.bandwidth(v)
            idx, kw = self.window(v, h)
            cf, _st = _local_fit(xs[idx] - v, kw, ws[idx], Ys[idx], deg, self._last_cf)
            self._last_cf = cf
            # subparcomp
            val = np.empty((2, ncol))
            if deg == 0:
                val[0] = cf[0] - (0.0 + pc_cf[0] * 1.0)
                val[1] = 0.0
            else:
                val[0] = cf[0] - ((0.0 + pc_cf[0] * 1.0) + pc_cf[1] * (v - xbar))
                val[1] = cf[1] - ((0.0 + pc_cf[0] * 0.0) + pc_cf[1] * 1.0)
            return h, val

        ll, ur = xs[0], xs[-1]
        for v in (ll, ur):
            h, val = fit_vertex(v)
            xv.append(v); hv.append(h); pseudo.append(False); vals.append(val)
            lo_v.append(-1); hi_v.append(-1)
        flo, fhi = ll, ur

        def split_score(i0, i1, a, b):
            hmin = 0.0
            for h in (hv[i0], hv[i1]):
                if h > 0 and (hmin == 0 or h < hmin):
                    hmin = h
            le = (b - a) / 1.0
            if hmin == 0:
                score = 2 * (b - a) / (fhi - flo) if fhi != flo else float("nan")
            else:
                score = le / hmin
            return le, (_CUT < score)

        # tree nodes: [i0, i1, left_child, right_child, mid_vertex]
        nodes = []
        midx = {}

        def grow(i0, i1, a, b):
            node = len(nodes)
            nodes.append([i0, i1, -1, -1, -1])
            le, split = split_score(i0, i1, a, b)
            if not split:
                return node
            key = (min(i0, i1), max(i0, i1))
            if key in midx:
                nv = midx[key]
            else:
                if len(xv) == self.nvm:
                    raise RuntimeError("newsplit: out of vertex space")
                nv = len(xv)
                xm = (xv[key[0]] + xv[key[1]]) / 2
                pv = le < _CUT * min(hv[i0], hv[i1])
                xv.append(xm); lo_v.append(key[0]); hi_v.append(key[1])
                if pv:  # pseudo vertex: values from parent cell (exvvalpv)
                    hv.append((hv[key[0]] + hv[key[1]]) / 2)
                    pseudo.append(True)
                    vl, vr = vals[i0], vals[i1]
                    dl = xv[i1] - xv[i0]
                    val = np.empty((2, ncol))
                    if deg == 0:
                        val[0] = (vl[0] + vr[0]) / 2
                        val[1] = 0.0
                    else:
                        val[0] = (vl[0] + vr[0]) / 2 + dl * (vl[1] - vr[1]) / 8
                        val[1] = 1.5 * (vr[0] - vl[0]) / dl - (vl[1] + vr[1]) / 4
                    vals.append(val)
                else:
                    hv.append(0.0); pseudo.append(False); vals.append(None)
                    h, val = fit_vertex(xm)
                    hv[nv] = h; vals[nv] = val
                midx[key] = nv
            nodes[node][4] = nv
            mid = (b + a) / 2
            left = grow(i0, nv, a, mid)
            right = grow(nv, i1, mid, b)
            nodes[node][2] = left
            nodes[node][3] = right
            return node

        try:
            grow(0, 1, ll, ur)
        except RecursionError:  # R overflows its C stack here as well
            raise RuntimeError("locfit: evaluation tree too deep (R fails with a C stack overflow)") from None

        # ---- fitted(): descend tree for every data point and interpolate
        xv_a = np.asarray(xv)
        V0 = np.stack([v[0] for v in vals])  # (nv, ncol)
        V1 = np.stack([v[1] for v in vals])
        nd = np.asarray(nodes, dtype=np.int64)
        cur = np.zeros(n, dtype=np.int64)
        active = np.flatnonzero(nd[cur, 2] >= 0)
        while active.size:
            c = cur[active]
            a = xv_a[nd[c, 0]]
            b = xv_a[nd[c, 1]]
            go_lo = (2 * (xs[active] - a)) < (b - a)
            cur[active] = np.where(go_lo, nd[c, 2], nd[c, 3])
            active = active[nd[cur[active], 2] >= 0]
        i0 = nd[cur, 0]
        i1 = nd[cur, 1]
        a = xv_a[i0]
        b = xv_a[i1]
        hx = xs - a
        dd = b - a
        f0, f1v = V0[i0], V0[i1]
        if deg == 0:
            # linear_interp(h, d, f0, f1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = ((dd - hx)[:, None] * f0 + hx[:, None] * f1v) / dd[:, None]
            z = dd == 0
            if z.any():
                out[z] = f0[z]
            pcx = 0.0 + pc_cf[0][None, :] * 1.0
        else:
            d0, d1 = V1[i0], V1[i1]
            with np.errstate(invalid="ignore", divide="ignore"):
                t = hx / dd
            ph1 = t * t * (3 - 2 * t)
            ph0 = 1 - ph1
            ph2 = t * (1 - t) * (1 - t)
            ph3 = t * t * (t - 1)
            z = dd == 0
            # hermite2 special cases (t<0 / t>1 cannot occur for data in cell)
            ph0 = np.where(z, 1.0, ph0); ph1 = np.where(z, 0.0, ph1)
            ph2 = np.where(z, 0.0, ph2); ph3 = np.where(z, 0.0, ph3)
            neg = (~z) & (t < 0)
            pos = (~z) & (t > 1)
            ph0 = np.where(neg, 1.0, np.where(pos, 0.0, ph0))
            ph1 = np.where(neg, 0.0, np.where(pos, 1.0, ph1))
            ph2 = np.where(neg, t, np.where(pos, 0.0, ph2))
            ph3 = np.where(neg, 0.0, np.where(pos, t - 1, ph3))
            ph2 = ph2 * dd
            ph3 = ph3 * dd
            out = (ph0[:, None] * f0 + ph1[:, None] * f1v) + ph2[:, None] * d0 + ph3[:, None] * d1
            pcx = (0.0 + pc_cf[0][None, :] * 1.0) + pc_cf[1][None, :] * (xs - xbar)[:, None]
        out = out + pcx
        res = np.empty_like(out)
        res[self.order] = out
        self.n_vertices = len(xv)
        return res


def locfit_by_col(y, x=None, weights=None, span=0.5, degree=0):
    """Port of ``edgeR::locfitByCol``.

    Smooths each column of ``y`` against covariate ``x`` with
    ``fitted(locfit(y[,j] ~ x, weights=weights, alpha=span, deg=degree))``.

    Parameters
    ----------
    y : array_like, (ntags,) or (ntags, ncol)
    x : array_like, (ntags,), optional. Defaults to ``1..ntags``.
    weights : scalar or array_like, optional. Prior weights (recycled as
        ``rep_len``); default 1.
    span : float. locfit nearest-neighbour fraction ``alpha``.
    degree : int, 0 or 1.

    Returns
    -------
    np.ndarray of shape (ntags, ncol) (a 1-D ``y`` is returned as a
    one-column matrix, as ``as.matrix`` would in R).
    """
    y = np.array(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    ntags = y.shape[0]
    if weights is None:
        weights = 1.0
    w = np.resize(np.asarray(weights, dtype=float).ravel(), ntags)
    if x is None:
        x = np.arange(1, ntags + 1, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    if span * ntags < 2 or ntags <= 1:
        return y
    if degree not in (0, 1):
        raise ValueError("only degree 0 or 1 is supported")
    if x.size != ntags:
        raise ValueError("x must have one value per row of y")
    if y.shape[1] == 0:
        return y
    lf = _Locfit1D(x, w, span, degree)
    return lf.fit(y)
