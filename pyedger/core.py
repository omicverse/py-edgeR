from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.stats import rankdata
from statsmodels.nonparametric.smoothers_lowess import lowess


def _as_weight_matrix(weights, shape: tuple[int, int], row_index=None, col_index=None) -> pd.DataFrame:
    if weights is None:
        return None
    if isinstance(weights, pd.DataFrame):
        arr = weights.to_numpy(dtype=float)
        index = weights.index if weights.shape[0] == shape[0] else row_index
        columns = weights.columns if weights.shape[1] == shape[1] else col_index
    else:
        arr = np.asarray(weights, dtype=float)
        index = row_index
        columns = col_index
    if arr.ndim == 0:
        arr = np.full(shape, float(arr))
    elif arr.ndim == 1:
        if arr.size == shape[1]:
            arr = np.broadcast_to(arr[None, :], shape).copy()
        elif arr.size == shape[0]:
            arr = np.broadcast_to(arr[:, None], shape).copy()
        elif arr.size == np.prod(shape):
            arr = arr.reshape(shape)
        else:
            raise ValueError("one-dimensional weights must match genes, samples, or counts.size.")
    elif arr.shape == (shape[1], shape[0]):
        arr = arr.T
    elif arr.shape != shape:
        raise ValueError(f"weights shape {arr.shape} is incompatible with counts shape {shape}.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("weights must be finite and non-negative.")
    return pd.DataFrame(arr, index=row_index, columns=col_index)


@dataclass
class DGEList:
    """Minimal Python equivalent of edgeR's ``DGEList``.

    ``counts`` is genes x samples. ``samples`` contains at least
    ``lib.size`` and ``norm.factors``.
    """

    counts: pd.DataFrame
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    genes: pd.DataFrame | None = None

    common_dispersion: float | None = None
    trended_dispersion: np.ndarray | None = None
    tagwise_dispersion: np.ndarray | None = None
    AveLogCPM: np.ndarray | None = None
    prior_df: float | None = None
    prior_n: float | None = None
    pseudo_counts: pd.DataFrame | None = None
    pseudo_lib_size: float | None = None
    weights: pd.DataFrame | None = None

    def __init__(self, counts, lib_size=None, norm_factors=None, samples=None, genes=None, group=None, weights=None):
        counts_df = counts.copy() if isinstance(counts, pd.DataFrame) else pd.DataFrame(np.asarray(counts, dtype=float))
        if (counts_df.to_numpy(dtype=float) < 0).any():
            raise ValueError("Negative counts are not permitted.")
        self.counts = counts_df
        n = counts_df.shape[1]
        lib = np.asarray(lib_size if lib_size is not None else counts_df.sum(axis=0), dtype=float)
        if lib.size != n:
            raise ValueError("lib_size length must match number of samples.")
        nf = np.asarray(norm_factors if norm_factors is not None else np.ones(n), dtype=float)
        if nf.size != n:
            raise ValueError("norm_factors length must match number of samples.")
        if samples is None:
            self.samples = pd.DataFrame(index=counts_df.columns)
        else:
            self.samples = samples.copy()
        self.samples["lib.size"] = lib
        self.samples["norm.factors"] = nf
        if group is not None:
            # keep categorical level order (R factor semantics); plain values stay as given
            self.samples["group"] = pd.Categorical(group) if isinstance(group, pd.Categorical) else list(group)
        self.genes = genes
        self.weights = None if weights is None else _as_weight_matrix(weights, counts_df.shape, counts_df.index, counts_df.columns)
        self.common_dispersion = None
        self.trended_dispersion = None
        self.tagwise_dispersion = None
        self.AveLogCPM = None
        self.prior_df = None
        self.prior_n = None
        self.pseudo_counts = None
        self.pseudo_lib_size = None


@dataclass
class DGEExact:
    table: pd.DataFrame
    comparison: tuple[str, str]
    genes: pd.DataFrame | None = None


@dataclass
class DGEGLM:
    coefficients: np.ndarray
    fitted_values: np.ndarray
    deviance: np.ndarray
    counts: pd.DataFrame
    design: np.ndarray
    offset: np.ndarray
    dispersion: np.ndarray | float
    df_residual: np.ndarray
    coef_names: list[str]
    samples: pd.DataFrame | None = None
    genes: pd.DataFrame | None = None
    AveLogCPM: np.ndarray | None = None
    prior_count: float = 0.0
    method: str = "scipy"
    df_prior: float | np.ndarray | None = None
    s2_prior: float | np.ndarray | None = None
    s2_post: np.ndarray | None = None
    df_residual_zeros: np.ndarray | None = None
    df_residual_adj: np.ndarray | None = None
    deviance_adj: np.ndarray | None = None
    unshrunk_coefficients: np.ndarray | None = None
    average_ql_dispersion: float | None = None
    top_proportion: float | None = None
    weights: np.ndarray | None = None
    leverage: np.ndarray | None = None
    unit_deviance_adj: np.ndarray | None = None
    unit_df_adj: np.ndarray | None = None


@dataclass
class DGELRT:
    table: pd.DataFrame
    comparison: str
    df_test: np.ndarray
    genes: pd.DataFrame | None = None


@dataclass
class TestResults:
    calls: pd.DataFrame
    levels: tuple[int, ...]
    labels: tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        return self.calls.copy()


def calc_norm_factors(
    object,
    lib_size=None,
    method: str = "TMM",
    ref_column: int | None = None,
    logratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    do_weighting: bool = True,
    Acutoff: float = -1e10,
    p: float = 0.75,
):
    """edgeR-compatible library normalization factors.

    For a ``DGEList`` input the object is mutated and returned. For a count
    matrix input, the numeric factor vector is returned, following edgeR's
    default method behavior.
    """

    if isinstance(object, DGEList):
        f = calc_norm_factors(
            object.counts,
            lib_size=object.samples["lib.size"].to_numpy(dtype=float),
            method=method,
            ref_column=ref_column,
            logratio_trim=logratio_trim,
            sum_trim=sum_trim,
            do_weighting=do_weighting,
            Acutoff=Acutoff,
            p=p,
        )
        object.samples["norm.factors"] = f
        return object

    x = object.to_numpy(dtype=float) if isinstance(object, pd.DataFrame) else np.asarray(object, dtype=float)
    if np.isnan(x).any():
        raise ValueError("NA counts not permitted.")
    if x.ndim != 2:
        raise ValueError("counts must be a two-dimensional matrix.")
    nsamples = x.shape[1]
    lib = np.asarray(lib_size if lib_size is not None else x.sum(axis=0), dtype=float)
    if lib.size != nsamples:
        lib = np.resize(lib, nsamples)

    method = "TMMwsp" if method == "TMMwzp" else method
    method = method.upper() if method != "TMMwsp" else method
    keep = (x > 0).sum(axis=1) != 0
    x = x[keep]
    if x.shape[0] == 0 or nsamples == 1:
        method = "NONE"

    if method == "NONE":
        f = np.ones(nsamples)
    elif method == "RLE":
        f = _calc_factor_rle(x) / lib
    elif method == "UPPERQUARTILE":
        f = _calc_factor_quantile(x, lib, p=p)
    elif method == "TMM":
        if ref_column is None:
            f75 = _calc_factor_quantile(x, lib, p=0.75)
            ref_column = int(np.argmax(np.sum(np.sqrt(x), axis=0)) if np.median(f75) < 1e-20 else np.argmin(np.abs(f75 - np.mean(f75))))
        f = np.array([
            _calc_factor_tmm(
                x[:, i],
                x[:, ref_column],
                lib[i],
                lib[ref_column],
                logratio_trim,
                sum_trim,
                do_weighting,
                Acutoff,
            )
            for i in range(nsamples)
        ])
    else:
        raise ValueError("method must be one of TMM, RLE, upperquartile, none.")

    with np.errstate(divide="ignore", invalid="ignore"):
        f = f / np.exp(np.mean(np.log(f)))
    return f


def _calc_factor_rle(data: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        gm = np.exp(np.mean(np.log(data), axis=1))
        ratios = data / gm[:, None]
    return np.array([np.median(ratios[gm > 0, j]) for j in range(data.shape[1])])


def _calc_factor_quantile(data: np.ndarray, lib_size: np.ndarray, p: float = 0.75) -> np.ndarray:
    # R quantile default type=7 is NumPy's default linear method.
    return np.quantile(data, p, axis=0) / lib_size


def _log2_libm(x) -> np.ndarray:
    """log2 via the C library, as R computes it.

    NumPy's SIMD log2 differs from libm by an ulp on some inputs. TMM trims on
    ranks of log-ratios that are heavily tied (small integer counts), so an ulp
    breaks or creates ties at the trim boundary and moves the factor by ~1e-3.
    The ratios take few distinct values, so evaluating libm on the unique ones
    is cheap.
    """
    x = np.asarray(x, dtype=float)
    u, inv = np.unique(x, return_inverse=True)
    out = np.empty(u.size)
    for i, t in enumerate(u.tolist()):
        out[i] = math.log2(t) if t > 0 else (-math.inf if t == 0 else math.nan)
    return out[inv.reshape(x.shape)]


def _calc_factor_tmm(
    obs,
    ref,
    libsize_obs,
    libsize_ref,
    logratio_trim,
    sum_trim,
    do_weighting,
    Acutoff,
) -> float:
    obs = np.asarray(obs, dtype=float)
    ref = np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logR = _log2_libm((obs / libsize_obs) / (ref / libsize_ref))
        absE = (_log2_libm(obs / libsize_obs) + _log2_libm(ref / libsize_ref)) / 2.0
        v = (libsize_obs - obs) / libsize_obs / obs + (libsize_ref - ref) / libsize_ref / ref
    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > Acutoff)
    logR = logR[fin]
    absE = absE[fin]
    v = v[fin]
    if logR.size == 0 or np.max(np.abs(logR)) < 1e-6:
        return 1.0
    n = logR.size
    loL = np.floor(n * logratio_trim) + 1
    hiL = n + 1 - loL
    loS = np.floor(n * sum_trim) + 1
    hiS = n + 1 - loS
    keep = (
        (rankdata(logR, method="average") >= loL)
        & (rankdata(logR, method="average") <= hiL)
        & (rankdata(absE, method="average") >= loS)
        & (rankdata(absE, method="average") <= hiS)
    )
    if do_weighting:
        denom = np.sum(1.0 / v[keep])
        f = np.nan if denom == 0 else np.sum(logR[keep] / v[keep]) / denom
    else:
        f = np.mean(logR[keep])
    if not np.isfinite(f):
        f = 0.0
    return float(2.0 ** f)


def effective_lib_sizes(x: DGEList) -> np.ndarray:
    return x.samples["lib.size"].to_numpy(dtype=float) * x.samples["norm.factors"].to_numpy(dtype=float)


def cpm(x, normalized_lib_sizes: bool = True, log: bool = False, prior_count: float = 2.0) -> pd.DataFrame:
    """Counts per million, matching edgeR's basic DGEList/matrix behavior."""

    if isinstance(x, DGEList):
        counts = x.counts
        lib = effective_lib_sizes(x) if normalized_lib_sizes else x.samples["lib.size"].to_numpy(dtype=float)
    else:
        counts = x.copy() if isinstance(x, pd.DataFrame) else pd.DataFrame(np.asarray(x, dtype=float))
        lib = counts.sum(axis=0).to_numpy(dtype=float)
    vals = counts.to_numpy(dtype=float)
    if log:
        scaled_prior = prior_count * lib / np.mean(lib)
        out = np.log2((vals + scaled_prior[None, :]) / (lib[None, :] + 2.0 * scaled_prior[None, :]) * 1e6)
    else:
        out = vals / lib[None, :] * 1e6
    return pd.DataFrame(out, index=counts.index, columns=counts.columns)


def filter_by_expr(
    y,
    design=None,
    group=None,
    lib_size=None,
    min_count: float = 10.0,
    min_total_count: float = 15.0,
    large_n: float = 10.0,
    min_prop: float = 0.7,
) -> pd.Series:
    """edgeR-style expression filter for count matrices or DGEList objects."""

    if isinstance(y, DGEList):
        counts = y.counts
        if design is None and group is None:
            design = getattr(y, "design", None)
            if design is None and "group" in y.samples.columns:
                group = y.samples["group"]
        if lib_size is None:
            lib_size = effective_lib_sizes(y)
    else:
        counts = y.copy() if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y, dtype=float))
    vals = counts.to_numpy(dtype=float)
    if lib_size is None:
        lib = vals.sum(axis=0)
    else:
        lib = np.asarray(lib_size, dtype=float)
        if lib.size != vals.shape[1]:
            raise ValueError("lib_size length must match number of samples.")

    if group is None:
        if design is None:
            min_sample_size = vals.shape[1]
        else:
            X = design.to_numpy(dtype=float) if isinstance(design, pd.DataFrame) else np.asarray(design, dtype=float)
            if X.shape[0] != vals.shape[1]:
                raise ValueError("design rows must match number of samples.")
            h = np.sum(X * (X @ np.linalg.pinv(X.T @ X)), axis=1)
            min_sample_size = 1.0 / np.max(h)
    else:
        cats = pd.Categorical(group)
        counts_by_group = np.asarray(pd.Series(cats).value_counts(sort=False), dtype=float)
        counts_by_group = counts_by_group[counts_by_group > 0]
        min_sample_size = float(np.min(counts_by_group))

    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * min_prop
    cpm_cutoff = float(min_count) / np.median(lib) * 1e6
    cpm_vals = vals / lib[None, :] * 1e6
    tol = 1e-14
    keep_cpm = np.sum(cpm_vals >= cpm_cutoff, axis=1) >= (min_sample_size - tol)
    keep_total = np.sum(vals, axis=1) >= (min_total_count - tol)
    return pd.Series(keep_cpm & keep_total, index=counts.index, name="keep")


def equalize_lib_sizes(object, group=None, dispersion=None, lib_size=None):
    """edgeR ``equalizeLibSizes`` quantile-to-quantile pseudo-count transform."""

    if isinstance(object, DGEList):
        groups = _get_group(object, group)
        disp = _resolve_dispersion(object, dispersion if dispersion is not None else "auto")
        lib = effective_lib_sizes(object)
        pseudo, pseudo_lib = equalize_lib_sizes(
            object.counts,
            group=groups,
            dispersion=disp,
            lib_size=lib,
        )
        object.pseudo_counts = pd.DataFrame(pseudo, index=object.counts.index, columns=object.counts.columns)
        object.pseudo_lib_size = pseudo_lib
        return object

    y = object.to_numpy(dtype=float) if isinstance(object, pd.DataFrame) else np.asarray(object, dtype=float)
    if y.ndim != 2:
        raise ValueError("counts must be a two-dimensional matrix.")
    n_tags, n_libs = y.shape
    groups = np.asarray(group if group is not None else np.ones(n_libs), dtype=np.object_)
    if groups.size != n_libs:
        raise ValueError("group length must match number of samples.")
    lib = np.asarray(lib_size if lib_size is not None else y.sum(axis=0), dtype=float)
    if lib.size != n_libs:
        raise ValueError("lib_size length must match number of samples.")
    if dispersion is None:
        dispersion = 0.05
    disp = np.asarray(dispersion, dtype=float)
    if disp.ndim == 0:
        disp = np.full(n_tags, float(disp))

    common_lib = float(np.exp(np.mean(np.log(lib))))
    input_mean = np.zeros_like(y, dtype=float)
    output_mean = np.zeros_like(y, dtype=float)
    for lev in pd.unique(groups):
        j = groups == lev
        beta = _one_group_abundance(y[:, j], np.log(lib[j]), disp)
        lam = np.exp(beta)
        input_mean[:, j] = lam[:, None] * lib[j][None, :]
        output_mean[:, j] = lam[:, None] * common_lib
    pseudo = q2qnbinom(y, input_mean=input_mean, output_mean=output_mean, dispersion=disp)
    pseudo[pseudo < 0] = 0
    return pseudo, common_lib


def ave_log_cpm(x, normalized_lib_sizes: bool = True, prior_count: float = 2.0, dispersion=None,
                lib_size=None, offset=None, weights=None) -> np.ndarray:
    """edgeR ``aveLogCPM``: one-group NB fit on prior-augmented counts (R_ave_log_cpm.cpp).

    For a DGEList, ``dispersion=None`` uses the common dispersion when present
    (as ``aveLogCPM.DGEList`` does); otherwise 0.05.
    """
    from . import _glm_batch as gb
    if isinstance(x, DGEList):
        counts = x.counts.to_numpy(dtype=float)
        if lib_size is None and offset is None:
            lib_size = effective_lib_sizes(x) if normalized_lib_sizes else x.samples["lib.size"].to_numpy(dtype=float)
        if dispersion is None:
            dispersion = x.common_dispersion
        if weights is None and x.weights is not None:
            weights = x.weights.to_numpy(dtype=float)
    else:
        counts = x.to_numpy(dtype=float) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
    n, s = counts.shape
    if n == 0:
        return np.zeros(0)
    if offset is None:
        lib = np.asarray(lib_size if lib_size is not None else counts.sum(axis=0), dtype=float)
        if np.all(counts == 0) and np.max(lib) == 0:
            return np.full(n, (-np.log(n) + np.log(1e6)) / np.log(2))
        off = np.broadcast_to(np.log(lib), counts.shape)
    else:
        off = np.asarray(offset, dtype=float)
        off = np.broadcast_to(off, counts.shape) if off.ndim <= 1 else off
    disp = np.asarray(0.05 if dispersion is None else dispersion, dtype=float)
    if np.all(np.isnan(disp)):
        disp = np.asarray(0.05)
    elif np.any(np.isnan(disp)):
        disp = np.where(np.isnan(disp), np.nanmean(disp), disp)
    phi = np.broadcast_to(disp[:, None] if disp.ndim == 1 else disp, counts.shape)
    W = np.ones_like(counts) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), counts.shape)
    libs = np.exp(off)
    prior = float(prior_count) * libs / libs.mean(axis=1, keepdims=True)
    beta = gb.mglm_one_group(counts + prior, np.log(libs + 2 * prior), phi, W, maxit=50, tol=1e-10)
    return (beta + np.log(1e6)) / np.log(2)


def estimate_common_disp(object, group=None, rowsum_filter: float = 5.0):
    """Estimate a common NB dispersion.

    This follows edgeR's classic ``estimateCommonDisp`` path: iteratively
    equalize library sizes and maximize the summed conditional likelihood in
    ``delta = dispersion / (1 + dispersion)``.
    """

    if isinstance(object, DGEList):
        y = object.counts.to_numpy(dtype=float)
        groups = _get_group(object, group)
        lib = effective_lib_sizes(object)
    else:
        y = object.to_numpy(dtype=float) if isinstance(object, pd.DataFrame) else np.asarray(object, dtype=float)
        groups = np.asarray(group if group is not None else np.ones(y.shape[1]), dtype=str)
        lib = y.sum(axis=0)
    if np.all(pd.Series(groups).value_counts().to_numpy() <= 1):
        disp = np.nan
        if isinstance(object, DGEList):
            object.common_dispersion = disp
            return object
        return disp
    keep = y.sum(axis=1) > rowsum_filter
    if not np.any(keep):
        raise ValueError("No genes satisfy rowsum filter.")
    else:
        disp = 0.01
        for _ in range(2):
            pseudo, _ = equalize_lib_sizes(y, group=groups, dispersion=disp, lib_size=lib)
            split = _split_into_groups(pseudo[keep], groups)

            def objective(delta):
                return -_common_cond_loglik_delta(split, delta)

            delta = _brent_fmin(objective, 1e-4, 100.0 / 101.0, 1e-6)
            disp = delta / (1.0 - delta)
    if isinstance(object, DGEList):
        object.common_dispersion = disp
        if np.isfinite(disp):
            object = equalize_lib_sizes(object, group=groups, dispersion=disp)
        object.AveLogCPM = ave_log_cpm(object, dispersion=disp)
        return object
    return disp


def estimate_tagwise_disp(object, prior_df: float = 10.0, group=None):
    """Estimate shrunken tagwise dispersions around ``common_dispersion``."""

    if not isinstance(object, DGEList):
        raise TypeError("estimate_tagwise_disp currently expects a DGEList.")
    if object.common_dispersion is None:
        estimate_common_disp(object, group=group)
    y = object.counts.to_numpy(dtype=float)
    groups = _get_group(object, group)
    lib = effective_lib_sizes(object)
    norm = y / lib[None, :] * np.mean(lib)
    levels = pd.unique(groups)
    raw = np.full(y.shape[0], object.common_dispersion if object.common_dispersion is not None else 0.0)
    for i in range(y.shape[0]):
        vals = []
        mus = []
        for lev in levels:
            z = norm[i, groups == lev]
            if z.size > 1:
                vals.append(np.var(z, ddof=1))
                mus.append(np.mean(z))
        if vals:
            mu = np.mean(mus)
            var = np.mean(vals)
            raw[i] = max((var - mu) / max(mu * mu, 1e-12), 0.0)
    nlibs = y.shape[1]
    ngroups = len(levels)
    prior_n = prior_df / max(nlibs - ngroups, 1)
    tagwise = (raw + prior_n * object.common_dispersion) / (1.0 + prior_n)
    object.tagwise_dispersion = tagwise
    object.trended_dispersion = np.full(y.shape[0], object.common_dispersion)
    object.prior_df = prior_df
    object.prior_n = prior_n
    object.AveLogCPM = ave_log_cpm(object, dispersion=tagwise)
    return object


def estimate_disp(
    object,
    group=None,
    tagwise: bool = True,
    prior_df: float | None = None,
    trend_method: str = "locfit",
    min_row_sum: float = 5.0,
    grid_length: int = 21,
    grid_range: tuple[float, float] = (-10.0, 10.0),
    span: float | None = None,
    legacy_span: bool = False,
    design=None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
):
    """Classic ``estimateDisp`` workflow.

    Current coverage targets the classic no-design path with
    ``trend_method`` values ``'none'``, ``'movingave'``, ``'loess'`` and
    ``'locfit'``. It uses edgeR's conditional likelihood grid and
    weighted-likelihood empirical Bayes shrinkage for tagwise dispersions. GLM
    design-based dispersion estimation is implemented for ``trend_method='none'``.
    """

    if not isinstance(object, DGEList):
        object = DGEList(object, group=group)
    trend_method = trend_method.lower()
    if trend_method not in {"none", "movingave", "loess", "locfit", "locfit.mixed"}:
        raise NotImplementedError("Only trend_method='none', 'movingave', 'loess', 'locfit' and 'locfit.mixed' are implemented for estimate_disp.")

    groups = _get_group(object, group)
    lib = effective_lib_sizes(object)
    y = object.counts.to_numpy(dtype=float)
    if design is None:
        if np.all(pd.Series(groups).value_counts().to_numpy() <= 1):
            object.common_dispersion = object.trended_dispersion = object.tagwise_dispersion = np.nan
            return object
        sel0 = y.sum(axis=1) >= min_row_sum
        pseudo0, _ = equalize_lib_sizes(y, group=groups, dispersion=0.01, lib_size=lib)
        split0 = _split_into_groups(pseudo0[sel0], groups)
        delta0 = _brent_fmin(lambda d: -_common_cond_loglik_delta(split0, d), 1e-4, 100.0 / 101.0, 1e-6)
        object.common_dispersion = delta0 / (1.0 - delta0)
        design_matrix = None
    elif isinstance(design, pd.DataFrame):
        design_matrix = design.to_numpy(dtype=float)
        if design_matrix.shape[0] != y.shape[1]:
            raise ValueError("design rows must match number of samples.")
        object.design = design
    else:
        design_matrix = np.asarray(design, dtype=float)
        if design_matrix.shape[0] != y.shape[1]:
            raise ValueError("design rows must match number of samples.")
        object.design = design_matrix
    ave = ave_log_cpm(object, dispersion=object.common_dispersion if object.common_dispersion is not None else 0.05)
    object.AveLogCPM = ave

    estimate_prior_df = prior_df is None
    sel = y.sum(axis=1) >= min_row_sum
    if not np.any(sel):
        object.tagwise_dispersion = np.full(y.shape[0], object.common_dispersion)
        object.prior_df = prior_df
        object.prior_n = np.nan
        return object

    if design_matrix is None:
        theta, l0 = _classic_cond_loglik_grid(
            y,
            groups=groups,
            lib_size=lib,
            equalize_dispersion=object.common_dispersion,
            sel=sel,
            grid_length=grid_length,
            grid_range=grid_range,
        )
        ncoefs = len(pd.unique(groups))
    else:
        theta, l0 = _design_adjusted_profile_loglik_grid(
            y[sel],
            design=design_matrix,
            offset=np.log(lib),
            weights=None if object.weights is None else object.weights.to_numpy(dtype=float)[sel],
            grid_length=grid_length,
            grid_range=grid_range,
        )
        ncoefs = design_matrix.shape[1]
    common_theta = _maximize_interpolant(theta, np.sum(l0, axis=0)[None, :])[0]
    object.common_dispersion = float(0.1 * 2.0**common_theta)
    # edgeR recomputes AveLogCPM at the common dispersion before fitting the trend.
    ave = ave_log_cpm(object.counts, lib_size=lib, dispersion=object.common_dispersion,
                      weights=None if object.weights is None else object.weights.to_numpy(dtype=float))
    if not tagwise or not np.isfinite(object.common_dispersion):
        object.trended_dispersion = None if trend_method == "none" else np.full(y.shape[0], object.common_dispersion)
        return object
    if trend_method in {"movingave", "loess", "locfit", "locfit.mixed"}:
        if span is None:
            nsel = int(np.sum(sel))   # WLEB default span (edgeR >= 4.8; legacy_span: edgeR <= 4.6)
            span = (_choose_lowess_span(nsel, small_n=50, min_span=0.25, power=0.5) if legacy_span
                    else _choose_lowess_span(nsel))
        if trend_method == "movingave":
            m0 = _moving_average_loglik(l0, ave[sel], span=float(span))
        elif trend_method == "locfit":
            m0 = _locfit_by_col(l0, ave[sel], span=float(span), degree=0)
        elif trend_method == "locfit.mixed":
            deg0 = _locfit_by_col(l0, ave[sel], span=float(span), degree=0)
            deg1 = _locfit_by_col(l0, ave[sel], span=float(span), degree=1)
            xsel = ave[sel]
            lo, hi = float(np.min(xsel)), float(np.max(xsel))
            if hi > lo:
                mix = stats.beta.cdf((xsel - lo) / (hi - lo), a=2, b=2)[:, None]
            else:
                mix = np.zeros((xsel.size, 1), dtype=float)
            m0 = mix * deg0 + (1.0 - mix) * deg1
            for j in range(m0.shape[1] - 1, 1, -1):
                diff1 = m0[:, j] - m0[:, j - 1]
                diff2 = m0[:, j - 1] - m0[:, j - 2]
                k = (diff1 > 0) & (diff2 < 0)
                if np.any(k):
                    m0[k, : j - 1] = m0[k, [j - 1]]
        else:
            m0 = _loess_by_col(l0, ave[sel], span=float(span))
        trend_theta = _maximize_interpolant(theta, m0)
        disp_trend_sel = 0.1 * 2.0**trend_theta
        trended_disp = np.full(y.shape[0], disp_trend_sel[np.argmin(ave[sel])])
        trended_disp[sel] = disp_trend_sel
        object.trended_dispersion = trended_disp
    else:
        span = None
        m0 = np.broadcast_to(np.mean(l0, axis=0)[None, :], l0.shape)
        object.trended_dispersion = None
    nlibs = y.shape[1]
    if estimate_prior_df:
        prior_df = _estimate_disp_prior_df(
            object,
            sel=sel,
            design=design_matrix,
            groups=groups,
            lib=lib,
            dispersion_trend=(object.trended_dispersion[sel] if object.trended_dispersion is not None else object.common_dispersion),
            covariate=(ave[sel] if trend_method != "none" else None),
            robust=robust,
            winsor_tail_p=winsor_tail_p,
        )
    prior_n = np.asarray(prior_df, dtype=float) / max(nlibs - ncoefs, 1)
    if trend_method in {"movingave", "loess", "locfit", "locfit.mixed"}:
        tagwise_disp = object.trended_dispersion.copy()
    else:
        tagwise_disp = np.full(y.shape[0], object.common_dispersion)
    # edgeR: near-infinite shrinkage (prior.n > 1e6, e.g. prior.df = Inf when
    # the data show no extra variability) leaves tagwise at the trend/common.
    prior_n_sel = prior_n[sel] if prior_n.ndim and prior_n.size == y.shape[0] else prior_n
    too_large = np.broadcast_to(prior_n_sel > 1e6, (int(np.sum(sel)),))
    if not np.all(too_large):
        temp_n = np.where(too_large, 1e6, prior_n_sel)
        individual = _maximize_interpolant(theta, l0 + temp_n[:, None] * m0)
        tagwise_sel = tagwise_disp[sel]
        upd = ~too_large if robust else np.ones_like(too_large)
        tagwise_sel[upd] = 0.1 * 2.0 ** individual[upd]
        tagwise_disp[sel] = tagwise_sel
    object.tagwise_dispersion = tagwise_disp
    object.prior_df = float(prior_df) if np.asarray(prior_df).ndim == 0 else np.asarray(prior_df, dtype=float)
    object.prior_n = float(prior_n) if np.asarray(prior_n).ndim == 0 else np.asarray(prior_n, dtype=float)
    object.span = span
    object.AveLogCPM = ave_log_cpm(object)      # aveLogCPM.DGEList: common dispersion
    return object


def estimate_glm_common_disp(object, design=None, method: str = "CoxReid", subset: int = 10000, **kwargs):
    """edgeR ``estimateGLMCommonDisp`` (CoxReid): maximise the summed Cox-Reid APL."""

    if method.lower() != "coxreid":
        raise NotImplementedError("Only method='CoxReid' is implemented for estimate_glm_common_disp.")
    if not isinstance(object, DGEList):
        object = DGEList(object)
    counts = object.counts.to_numpy(dtype=float)
    X = _default_design_from_group(object).to_numpy(dtype=float) if design is None and "group" in object.samples else (
        np.ones((counts.shape[1], 1)) if design is None else np.asarray(design, dtype=float))
    if X.shape[1] >= counts.shape[1]:
        object.common_dispersion = np.nan
        return object
    off = np.broadcast_to(np.log(effective_lib_sizes(object)), counts.shape)
    W = None if object.weights is None else object.weights.to_numpy(dtype=float)
    object.common_dispersion = _disp_cox_reid(counts, X, off, weights=W, ave_log_cpm=ave_log_cpm(object), subset=subset, **kwargs)
    return object


def _brent_fmin(f, ax, bx, tol):
    """R stats optimize.c Brent_fmin (minimiser)."""
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(np.finfo(float).eps)
    a, b = ax, bx
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fx = f(x); fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def _disp_cox_reid(y, design, offset, weights=None, ave_log_cpm=None, interval=(0.0, 4.0), tol=1e-5,
                   min_row_sum=5, subset=10000):
    """edgeR dispCoxReid."""
    from . import _glm_batch as gb
    y = np.asarray(y, dtype=float)
    offset = np.broadcast_to(np.asarray(offset, dtype=float), y.shape)
    W = np.ones_like(y) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), y.shape)
    keep = y.sum(axis=1) >= min_row_sum
    y, offset, W = y[keep], offset[keep], W[keep]
    if ave_log_cpm is not None:
        ave_log_cpm = np.asarray(ave_log_cpm, dtype=float)[keep]
    if y.shape[0] < 1:
        raise ValueError("no data rows with required number of counts")
    if subset is not None and subset <= y.shape[0] / 2:
        if ave_log_cpm is None:
            ave_log_cpm = globals()["ave_log_cpm"](y, offset=offset, weights=W)
        ratio = int(np.floor(y.shape[0] / subset))
        if ratio > 1:
            o = np.argsort(ave_log_cpm, kind="stable")
            i = o[np.arange(int(np.floor(ratio / 2)), y.shape[0], ratio)]
            y, offset, W = y[i], offset[i], W[i]
    X = np.asarray(design, dtype=float)

    def negapl(par):
        phi = np.full_like(y, par ** 4)
        _, mu, _ = _fit_glm_batch(y, X, offset, phi, W)
        return -np.sum(gb.compute_apl(y, mu, phi, W, X))

    return _brent_fmin(negapl, interval[0] ** 0.25, interval[1] ** 0.25, tol) ** 4


def estimate_glm_trended_disp(object, design=None, method: str = "auto", span: float | None = None, **kwargs):
    """Compatibility wrapper for edgeR's legacy GLM trended-dispersion API."""

    if not isinstance(object, DGEList):
        object = DGEList(object)
    method_l = method.lower()
    if method_l in {"auto", "bin.spline", "spline", "power"}:
        trend_method = "locfit"
    elif method_l == "bin.loess":
        trend_method = "loess"
    else:
        raise NotImplementedError("Unsupported GLM trend method.")
    estimate_disp(object, design=design, tagwise=False, trend_method=trend_method, span=span, **kwargs)
    if object.trended_dispersion is None:
        object.trended_dispersion = np.full(object.counts.shape[0], object.common_dispersion)
    object.trend_method = method
    return object


def estimate_glm_tagwise_disp(
    object,
    design=None,
    prior_df: float | None = 10.0,
    trend: bool | None = None,
    span: float | None = None,
    **kwargs,
):
    """Compatibility wrapper for edgeR's legacy GLM tagwise-dispersion API."""

    if not isinstance(object, DGEList):
        object = DGEList(object)
    if trend is None:
        trend = object.trended_dispersion is not None
    if trend and object.trended_dispersion is None:
        estimate_glm_trended_disp(object, design=design, span=span, **kwargs)
    elif object.common_dispersion is None:
        estimate_glm_common_disp(object, design=design, **kwargs)
    trend_method = "locfit" if trend else "none"
    estimate_disp(object, design=design, tagwise=True, prior_df=prior_df, trend_method=trend_method, span=span, **kwargs)
    return object


def glm_fit(
    y,
    design=None,
    dispersion=None,
    offset=None,
    lib_size=None,
    prior_count: float = 0.125,
    weights=None,
    start=None,
) -> DGEGLM:
    """Fit row-wise negative-binomial GLMs, mirroring ``edgeR::glmFit``.

    This path is intended for fixed-dispersion GLM workflows and currently
    supports dense designs and observation weights.
    """

    if isinstance(y, DGEList):
        obj = y
        counts_df = obj.counts
        if design is None:
            design = _default_design_from_group(obj)
        if dispersion is None:
            dispersion = _resolve_dispersion(obj, "auto")
        if offset is None:
            offset = np.log(effective_lib_sizes(obj))
        if weights is None:
            weights = obj.weights
        fit = glm_fit(
            counts_df,
            design=design,
            dispersion=dispersion,
            offset=offset,
            prior_count=prior_count,
            weights=weights,
            start=start,
        )
        fit.samples = obj.samples
        fit.genes = obj.genes
        fit.AveLogCPM = obj.AveLogCPM if obj.AveLogCPM is not None else ave_log_cpm(obj)
        return fit

    counts_df = y.copy() if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y, dtype=float))
    counts = counts_df.to_numpy(dtype=float)
    n_tags, n_libs = counts.shape
    if design is None:
        X = np.ones((n_libs, 1), dtype=float)
        coef_names = ["Intercept"]
    elif isinstance(design, pd.DataFrame):
        X = design.to_numpy(dtype=float)
        coef_names = list(design.columns.astype(str))
    else:
        X = np.asarray(design, dtype=float)
        coef_names = [f"coef_{i}" for i in range(X.shape[1])]
    if X.shape[0] != n_libs:
        raise ValueError("design rows must match count columns.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("design matrix is not full rank.")

    if dispersion is None:
        raise ValueError("No dispersion values provided.")
    disp = np.asarray(dispersion, dtype=float)
    if disp.ndim == 0:
        disp_vec = np.full(n_tags, float(disp))
    else:
        disp_vec = disp.reshape(-1)
        if disp_vec.size != n_tags:
            raise ValueError("dispersion length must be 1 or number of rows.")

    if offset is None:
        lib = np.asarray(lib_size if lib_size is not None else counts.sum(axis=0), dtype=float)
        off = np.log(lib)
    else:
        off = np.asarray(offset, dtype=float)
        if off.ndim == 0:
            off = np.full(n_libs, float(off))
    if off.ndim == 1:
        offset_mat = np.broadcast_to(off, counts.shape).copy()
    else:
        offset_mat = off.astype(float)
        if offset_mat.shape != counts.shape:
            raise ValueError("offset matrix must have same shape as counts.")
    weight_df = None if weights is None else _as_weight_matrix(weights, counts.shape, counts_df.index, counts_df.columns)
    weight_mat = None if weight_df is None else weight_df.to_numpy(dtype=float)

    W = np.ones_like(counts) if weight_mat is None else weight_mat
    phi_mat = np.broadcast_to(disp_vec[:, None], counts.shape)
    beta, fitted, dev = _fit_glm_batch(counts, X, offset_mat, phi_mat, W, start=start)

    unshrunk_beta = None
    if prior_count > 0:
        # edgeR predFC: refit on prior-augmented counts with the same algorithm.
        unshrunk_beta = beta.copy()
        aug_counts, aug_offset = _add_prior_count(counts, offset_mat, prior_count)
        beta, _, _ = _fit_glm_batch(aug_counts, X, aug_offset, phi_mat, W)

    return DGEGLM(
        coefficients=beta,
        fitted_values=fitted,
        deviance=dev,
        counts=counts_df,
        design=X,
        offset=offset_mat,
        dispersion=disp_vec if disp_vec.size > 1 else float(disp_vec[0]),
        df_residual=np.full(n_tags, n_libs - X.shape[1], dtype=float),
        coef_names=coef_names,
        AveLogCPM=ave_log_cpm(counts_df),
        prior_count=prior_count,
        unshrunk_coefficients=unshrunk_beta,
        weights=weight_mat,
    )


def _fit_glm_batch(counts, X, offset_mat, phi_mat, W, start=None):
    """edgeR glmFit.default core: one-way shortcut or Levenberg, all genes at once."""
    from . import _glm_batch as gb
    if gb.is_one_way(X):
        beta, mu = gb.mglm_one_way(counts, X, offset_mat, phi_mat, W, start=start)
        dev = gb.nb_deviance(counts, mu, phi_mat, W)
    else:
        if start is None:
            start = gb.levenberg_start(counts, offset_mat, phi_mat, W, X)
        beta, mu, dev, _, _ = gb.fit_levenberg(counts, offset_mat, phi_mat, W, X, start, tol=1e-6, maxit=250)
    return beta, mu, dev


def pred_fc(y, design, prior_count: float = 0.125, offset=None, dispersion=None, weights=None) -> pd.DataFrame:
    """edgeR-style prior-count shrunken GLM coefficients on log2 scale."""

    if isinstance(y, DGEList):
        obj = y
        counts_df = obj.counts
        if offset is None:
            offset = np.log(effective_lib_sizes(obj))
        if dispersion is None:
            dispersion = _resolve_dispersion(obj, "auto") if (
                obj.common_dispersion is not None or obj.trended_dispersion is not None or obj.tagwise_dispersion is not None
            ) else 0.0
    else:
        counts_df = y.copy() if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y, dtype=float))
        if dispersion is None:
            dispersion = 0.0
    counts = counts_df.to_numpy(dtype=float)
    n_tags, n_libs = counts.shape
    if offset is None:
        off = np.log(counts.sum(axis=0))
    else:
        off = np.asarray(offset, dtype=float)
        if off.ndim == 0:
            off = np.full(n_libs, float(off))
    if off.ndim == 1:
        offset_mat = np.broadcast_to(off, counts.shape).copy()
    else:
        offset_mat = off.astype(float)
        if offset_mat.shape != counts.shape:
            raise ValueError("offset matrix must have same shape as counts.")
    aug_counts, aug_offset = _add_prior_count(counts, offset_mat, prior_count)
    aug_df = pd.DataFrame(aug_counts, index=counts_df.index, columns=counts_df.columns)
    fit = glm_fit(
        aug_df,
        design=design,
        dispersion=dispersion,
        offset=aug_offset,
        weights=weights,
        prior_count=0,
    )
    return pd.DataFrame(fit.coefficients / np.log(2.0), index=counts_df.index, columns=fit.coef_names)


def glm_lrt(glmfit: DGEGLM, coef=None, contrast=None) -> DGELRT:
    """Likelihood-ratio test for a fitted NB GLM."""

    if not isinstance(glmfit, DGEGLM):
        raise TypeError("glm_lrt expects a DGEGLM from glm_fit.")
    design = np.asarray(glmfit.design, dtype=float)
    if design.shape[1] < 2:
        raise ValueError("Need at least two design columns.")
    if contrast is not None:
        c = _as_contrast_vector(contrast, design.shape[1])
        logfc = (glmfit.coefficients @ c) / np.log(2.0)
        reform = _contrast_as_first_coef_design(design, c)
        design0 = reform[:, 1:]
        comparison = _contrast_name(c, glmfit.coef_names)
    else:
        if coef is None:
            coef_i = design.shape[1] - 1
        elif isinstance(coef, str):
            coef_i = glmfit.coef_names.index(coef)
        else:
            coef_i = int(coef)
            if coef_i >= 1:
                # Python API accepts zero-based indices; R-style callers often
                # pass 2 for the second coefficient, so tolerate that too.
                coef_i = coef_i if coef_i < design.shape[1] else coef_i - 1
        logfc = glmfit.coefficients[:, coef_i] / np.log(2.0)
        design0 = np.delete(design, coef_i, axis=1)
        comparison = glmfit.coef_names[coef_i]

    null_disp = glmfit.dispersion
    if glmfit.average_ql_dispersion is not None:      # new QL method fits the null at the working dispersion
        null_disp = np.asarray(glmfit.dispersion, dtype=float) / glmfit.average_ql_dispersion
    null = glm_fit(
        glmfit.counts,
        design=design0,
        dispersion=null_disp,
        offset=glmfit.offset,
        prior_count=0,
        weights=glmfit.weights,
    )
    lr = null.deviance - glmfit.deviance
    df_test = null.df_residual - glmfit.df_residual
    pvals = stats.chi2.sf(lr, df_test)
    table = pd.DataFrame(
        {
            "logFC": logfc,
            "logCPM": glmfit.AveLogCPM if glmfit.AveLogCPM is not None else ave_log_cpm(glmfit.counts),
            "LR": lr,
            "PValue": pvals,
        },
        index=glmfit.counts.index,
    )
    return DGELRT(table=table, comparison=str(comparison), df_test=df_test, genes=glmfit.genes)


def _add_prior_count(counts: np.ndarray, offset: np.ndarray, prior_count: float):
    lib = np.exp(np.asarray(offset, dtype=float))
    row_mean = np.mean(lib, axis=1, keepdims=True)
    row_mean[row_mean <= 0] = 1.0
    prior = float(prior_count) * lib / row_mean
    return counts + prior, np.log(lib + 2.0 * prior)


def add_prior_count(y, lib_size=None, offset=None, prior_count: float = 1.0):
    """Add edgeR-style library-size-adjusted prior counts.

    Returns a dict with augmented count matrix ``y`` and adjusted log-offset
    matrix ``offset``, matching edgeR's public ``addPriorCount`` result shape.
    """

    if isinstance(y, DGEList):
        counts_df = y.counts
        if lib_size is None and offset is None:
            lib_size = effective_lib_sizes(y)
    else:
        counts_df = y.copy() if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y, dtype=float))
    counts = counts_df.to_numpy(dtype=float)
    n_tags, n_libs = counts.shape
    if offset is None:
        lib = np.asarray(lib_size if lib_size is not None else counts.sum(axis=0), dtype=float)
        if lib.size != n_libs:
            raise ValueError("lib_size length must match number of samples.")
        offset_mat = np.broadcast_to(np.log(lib)[None, :], counts.shape).copy()
    else:
        off = np.asarray(offset, dtype=float)
        if off.ndim == 0:
            off = np.full(n_libs, float(off))
        if off.ndim == 1:
            if off.size != n_libs:
                raise ValueError("offset length must match number of samples.")
            offset_mat = np.broadcast_to(off[None, :], counts.shape).copy()
        else:
            offset_mat = off.astype(float)
            if offset_mat.shape != counts.shape:
                raise ValueError("offset matrix must have same shape as counts.")
    aug_counts, aug_offset = _add_prior_count(counts, offset_mat, prior_count)
    return {
        "y": pd.DataFrame(aug_counts, index=counts_df.index, columns=counts_df.columns),
        "offset": pd.DataFrame(aug_offset, index=counts_df.index, columns=counts_df.columns),
    }


def glm_treat(glmfit: DGEGLM, coef=None, contrast=None, lfc: float = np.log2(1.2), null: str = "interval") -> DGELRT:
    """edgeR-style thresholded GLM likelihood-ratio test."""

    if lfc < 0:
        raise ValueError("lfc must be non-negative.")
    if lfc == 0:
        return glm_lrt(glmfit, coef=coef, contrast=contrast)
    if not isinstance(glmfit, DGEGLM):
        raise TypeError("glm_treat expects a DGEGLM from glm_fit.")
    is_lrt = glmfit.df_prior is None

    design = np.asarray(glmfit.design, dtype=float)
    if design.shape[1] < 2:
        raise ValueError("Need at least two design columns.")
    shrunk = glmfit.prior_count != 0 and glmfit.unshrunk_coefficients is not None
    if contrast is not None:
        c = _as_contrast_vector(contrast, design.shape[1])
        logfc = (glmfit.coefficients @ c) / np.log(2.0)
        unshrunk_logfc = ((glmfit.unshrunk_coefficients if shrunk else glmfit.coefficients) @ c) / np.log(2.0)
        design = _contrast_as_first_coef_design(design, c)
        coef_vec = design[:, 0]
        design0 = design[:, 1:]
        comparison = _contrast_name(c, glmfit.coef_names)
    else:
        if coef is None:
            coef_i = design.shape[1] - 1
        elif isinstance(coef, str):
            coef_i = glmfit.coef_names.index(coef)
        else:
            coef_i = int(coef)
            if coef_i >= 1:
                coef_i = coef_i if coef_i < design.shape[1] else coef_i - 1
        logfc = glmfit.coefficients[:, coef_i] / np.log(2.0)
        coef_source = glmfit.unshrunk_coefficients if shrunk else glmfit.coefficients
        unshrunk_logfc = coef_source[:, coef_i] / np.log(2.0)
        coef_vec = design[:, coef_i]
        design0 = np.delete(design, coef_i, axis=1)
        comparison = glmfit.coef_names[coef_i]

    offset_old = np.asarray(glmfit.offset, dtype=float)
    offset_adj = lfc * np.log(2.0) * coef_vec[None, :]
    dispersion = glmfit.dispersion
    if glmfit.average_ql_dispersion is not None:
        dispersion = np.asarray(dispersion, dtype=float) / float(glmfit.average_ql_dispersion)

    fit0_left = glm_fit(
        glmfit.counts,
        design=design0,
        offset=offset_old + offset_adj,
        weights=glmfit.weights,
        dispersion=dispersion,
        prior_count=0,
    )
    fit1_left = glm_fit(
        glmfit.counts,
        design=design,
        offset=offset_old + offset_adj,
        weights=glmfit.weights,
        dispersion=dispersion,
        prior_count=0,
    )
    z_left = np.sqrt(np.maximum(0.0, fit0_left.deviance - fit1_left.deviance))

    fit0_right = glm_fit(
        glmfit.counts,
        design=design0,
        offset=offset_old - offset_adj,
        weights=glmfit.weights,
        dispersion=dispersion,
        prior_count=0,
    )
    fit1_right = glm_fit(
        glmfit.counts,
        design=design,
        offset=offset_old - offset_adj,
        weights=glmfit.weights,
        dispersion=dispersion,
        prior_count=0,
    )
    z_right = np.sqrt(np.maximum(0.0, fit0_right.deviance - fit1_right.deviance))

    swap = z_left > z_right
    if np.any(swap):
        tmp = z_left[swap].copy()
        z_left[swap] = z_right[swap]
        z_right[swap] = tmp

    if not is_lrt:
        if glmfit.s2_post is None:
            raise ValueError("glm_treat for QL fits expects s2_post from glm_ql_fit.")
        if glmfit.df_residual_zeros is not None:
            df_resid = glmfit.df_residual_zeros
        elif glmfit.df_residual_adj is not None:
            df_resid = glmfit.df_residual_adj
        else:
            df_resid = glmfit.df_residual
        df_total = np.asarray(glmfit.df_prior, dtype=float) + df_resid
        max_df_resid = glmfit.counts.shape[1] - design.shape[1]
        df_total = np.minimum(df_total, glmfit.counts.shape[0] * max_df_resid)
        scale = np.sqrt(np.asarray(glmfit.s2_post, dtype=float))
        z_left = _zscore_t(z_left / scale, df_total)
        z_right = _zscore_t(z_right / scale, df_total)

    within = np.abs(unshrunk_logfc) <= lfc
    z_left = z_left * (2.0 * within.astype(float) - 1.0)

    null = null.lower()
    if null not in {"interval", "worst.case", "worst_case"}:
        raise ValueError("null must be 'interval' or 'worst.case'.")
    if null == "interval":
        c = 1.470402
        pvals = np.ones_like(z_left, dtype=float)
        use_tail = z_right + z_left > c
        pvals[use_tail] = _integrate_pnorm(-z_right[use_tail], -z_right[use_tail] + c) + _integrate_pnorm(
            z_left[use_tail] - c, z_left[use_tail]
        )
        pvals[~use_tail] = 2.0 * _integrate_pnorm(-z_right[~use_tail], z_left[~use_tail])
    else:
        pvals = stats.norm.cdf(-z_right) + stats.norm.cdf(z_left)

    if shrunk:
        table_data = {
            "logFC": logfc,
            "unshrunk.logFC": unshrunk_logfc,
            "logCPM": glmfit.AveLogCPM if glmfit.AveLogCPM is not None else ave_log_cpm(glmfit.counts),
            "PValue": pvals,
        }
    else:
        table_data = {
            "logFC": unshrunk_logfc,
            "logCPM": glmfit.AveLogCPM if glmfit.AveLogCPM is not None else ave_log_cpm(glmfit.counts),
            "PValue": pvals,
        }
    table = pd.DataFrame(table_data, index=glmfit.counts.index)
    return DGELRT(table=table, comparison=str(comparison), df_test=np.ones_like(pvals), genes=glmfit.genes)


def _as_contrast_vector(contrast, ncoef: int) -> np.ndarray:
    c = np.asarray(contrast, dtype=float)
    if c.ndim == 2:
        if c.shape[0] == ncoef:
            c = c[:, 0]
        elif c.shape[1] == ncoef:
            c = c[0, :]
        else:
            raise ValueError("contrast length must match number of coefficients.")
    else:
        c = c.reshape(-1)
    if c.size != ncoef:
        raise ValueError("contrast length must match number of coefficients.")
    if np.all(np.abs(c) < 1e-12):
        raise ValueError("contrast is all zero.")
    return c


def _contrast_as_first_coef_design(design: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Reparameterize design so the first coefficient is the requested contrast."""

    c = np.asarray(contrast, dtype=float).reshape(-1)
    a0 = c / float(c @ c)
    _, _, vh = np.linalg.svd(c[None, :], full_matrices=True)
    transform = np.column_stack([a0, vh[1:].T])
    return np.asarray(design, dtype=float) @ transform


def _contrast_name(contrast: np.ndarray, coef_names) -> str:
    parts = []
    for value, name in zip(contrast, coef_names):
        if abs(value) > 1e-12:
            parts.append(f"{value:g}*{name}")
    return " ".join(parts) if parts else "contrast"


def _integrate_pnorm(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=float)
    aa, bb = np.broadcast_arrays(a, b)
    same = aa == bb
    out[same] = stats.norm.cdf(aa[same])
    if np.any(~same):
        x0 = aa[~same]
        x1 = bb[~same]
        out[~same] = (x1 * stats.norm.cdf(x1) + stats.norm.pdf(x1) - (x0 * stats.norm.cdf(x0) + stats.norm.pdf(x0))) / (x1 - x0)
    return out


def _zscore_t(x, df):
    x = np.asarray(x, dtype=float)
    df = np.asarray(df, dtype=float)
    xx, dd = np.broadcast_arrays(x, df)
    tail = stats.t.logsf(np.abs(xx), dd)
    z = stats.norm.isf(np.exp(tail))
    return z * np.sign(xx)


def glm_ql_fit(
    y,
    design=None,
    dispersion=None,
    offset=None,
    lib_size=None,
    abundance_trend: bool = True,
    robust: bool = False,
    legacy: bool = True,
    top_proportion: float = 0.05,
    weights=None,
    ave_log_cpm=None,
) -> DGEGLM:
    """edgeR 4.0 ``glmQLFit``: quasi-likelihood NB GLM fit.

    ``legacy=True`` (the edgeR 4.0 default) is the classic QL method with
    zero-adjusted residual df. ``legacy=False`` uses the bias-adjusted
    deviance and residual df of Chen, Chen & Smyth (2024), which later edgeR
    releases make the default.
    """
    from . import _glm_batch as gb
    from ._ql_adjust import compute_adjust_s2, compute_prior_s2

    if robust:
        raise NotImplementedError("Robust QL moderation is not implemented yet.")
    samples = genes = None
    if isinstance(y, DGEList):
        obj = y
        if design is None:
            design = getattr(obj, "design", None)
            if design is None and "group" in obj.samples and pd.Series(obj.samples["group"]).nunique() > 1:
                design = _default_design_from_group(obj)
        if obj.AveLogCPM is None:
            obj.AveLogCPM = globals()["ave_log_cpm"](obj)
        if ave_log_cpm is None:
            ave_log_cpm = obj.AveLogCPM
        if legacy and dispersion is None:
            dispersion = obj.trended_dispersion if obj.trended_dispersion is not None else obj.common_dispersion
            if dispersion is None:
                raise ValueError("No dispersion values found in DGEList object.")
        if offset is None:
            offset = np.log(effective_lib_sizes(obj))
        if weights is None and obj.weights is not None:
            weights = obj.weights.to_numpy(dtype=float)
        samples, genes = obj.samples, obj.genes
        counts_df = obj.counts
    else:
        counts_df = y.copy() if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y, dtype=float))
    counts = counts_df.to_numpy(dtype=float)
    n, s_ = counts.shape
    if design is None:
        design = np.ones((s_, 1))
        coef_names = ["Intercept"]
    elif isinstance(design, pd.DataFrame):
        coef_names = list(design.columns.astype(str))
        design = design.to_numpy(dtype=float)
    else:
        design = np.asarray(design, dtype=float)
        coef_names = [f"coef_{i}" for i in range(design.shape[1])]
    if offset is None:
        lib = np.asarray(lib_size if lib_size is not None else counts.sum(axis=0), dtype=float)
        offset = np.log(lib)
    offset = np.asarray(offset, dtype=float)
    offset_mat = np.broadcast_to(offset, counts.shape).astype(float) if offset.ndim == 1 else offset.astype(float)
    W = np.ones_like(counts) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), counts.shape).astype(float)
    if ave_log_cpm is None:
        ave_log_cpm = globals()["ave_log_cpm"](counts, offset=offset_mat, weights=W, dispersion=dispersion)
    ave_log_cpm = np.asarray(ave_log_cpm, dtype=float)
    if dispersion is None:
        if legacy:
            raise ValueError("No dispersion values provided.")
        if top_proportion < 0 or top_proportion > 1:
            raise ValueError("top.proportion should be between 0 and 1.")
        top = ave_log_cpm >= np.quantile(ave_log_cpm, 1 - top_proportion, method="linear")
        sub = counts[top]
        dispersion = _disp_cox_reid(sub, design, np.log(sub.sum(axis=0)), weights=W[top])

    def fit_at(disp):
        return glm_fit(counts_df, design=design, dispersion=disp, offset=offset_mat, prior_count=0.125,
                       weights=None if weights is None else W)

    fit = fit_at(dispersion)
    fit.coef_names = coef_names
    fit.deviance = np.maximum(fit.deviance, 0.0)
    covariate = ave_log_cpm if abundance_trend else None
    if legacy:
        zerofit = (fit.fitted_values < 1e-4) & (counts < 1e-4)
        df = gb.resid_df(zerofit, design)
        fit.df_residual_zeros = df
        s2 = np.where(df == 0, 0.0, fit.deviance / np.where(df == 0, 1.0, df))
    else:
        disp = dispersion
        disp_vec = np.broadcast_to(np.asarray(disp, dtype=float), (n,)).astype(float)
        prior_disp = 1.0
        if weights is None:
            out = compute_adjust_s2(counts, fit.fitted_values, design, disp_vec, prior_disp, W)
            prior_disp = compute_prior_s2(out, covariate)
            out = compute_adjust_s2(counts, fit.fitted_values, design, disp_vec, prior_disp, W)
            prior_disp = compute_prior_s2(out, covariate)
            fit = fit_at(disp_vec / prior_disp)
            fit.coef_names = coef_names
            fit.deviance = np.maximum(fit.deviance, 0.0)
        fit.dispersion = disp
        out = compute_adjust_s2(counts, fit.fitted_values, design, disp_vec, prior_disp, W)
        fit.leverage = out["leverage"]
        fit.unit_deviance_adj = out["unit_deviance"]
        fit.unit_df_adj = out["unit_df"]
        s2 = out["s2"]
        df = out["df"]
        fit.df_residual_adj = df
        fit.deviance_adj = out["deviance"]
        fit.average_ql_dispersion = prior_disp
    df = np.where(df < 0.99, 0.0, df)
    df_prior, var_prior, var_post = _squeeze_var(s2, df, covariate=covariate)
    fit.df_prior = df_prior
    fit.s2_prior = var_prior
    fit.s2_post = var_post
    fit.AveLogCPM = ave_log_cpm
    fit.samples, fit.genes = samples, genes
    fit.top_proportion = top_proportion
    return fit


def glm_qlf_test(glmfit: DGEGLM, coef=None, contrast=None, poisson_bound: bool = True) -> DGELRT:
    if not isinstance(glmfit, DGEGLM) or glmfit.s2_post is None:
        raise ValueError("glm_qlf_test expects a DGEGLM produced by glm_ql_fit.")
    out = glm_lrt(glmfit, coef=coef, contrast=contrast)
    if glmfit.df_residual_zeros is not None:
        df_resid = glmfit.df_residual_zeros
    elif glmfit.df_residual_adj is not None:
        df_resid = glmfit.df_residual_adj
        poisson_bound = False
    else:
        df_resid = glmfit.df_residual
    f_stat = out.table["LR"].to_numpy(dtype=float) / out.df_test / glmfit.s2_post
    df_total = np.asarray(glmfit.df_prior, dtype=float) + df_resid
    df_total = np.minimum(df_total, np.sum(glmfit.df_residual))
    pvals = stats.f.sf(f_stat, out.df_test, df_total)
    if poisson_bound:
        # edgeR .isBelowPoissonBound: never more significant than the Poisson-variance LRT.
        disp = np.broadcast_to(np.asarray(glmfit.dispersion, dtype=float).reshape(-1, 1) if np.ndim(glmfit.dispersion) else glmfit.dispersion,
                               glmfit.fitted_values.shape)
        s2 = np.asarray(glmfit.s2_post, dtype=float).reshape(-1, 1)
        below = np.any((glmfit.fitted_values * disp + 1) * s2 < 1, axis=1)
        if np.any(below):
            i = np.flatnonzero(below)
            pois = glm_fit(glmfit.counts.iloc[i], design=glmfit.design, dispersion=0.0, offset=np.asarray(glmfit.offset)[i],
                           prior_count=0.125, weights=None if glmfit.weights is None else np.asarray(glmfit.weights)[i],
                           start=None if glmfit.unshrunk_coefficients is None else glmfit.unshrunk_coefficients[i])
            pois.coef_names = glmfit.coef_names
            pres = glm_lrt(pois, coef=coef, contrast=contrast)
            pvals[i] = np.maximum(pvals[i], pres.table["PValue"].to_numpy(dtype=float))
    out.table = out.table.drop(columns=["LR", "PValue"])
    out.table["F"] = f_stat
    out.table["PValue"] = pvals
    return out


def exact_test(
    object: DGEList,
    pair=None,
    dispersion="auto",
    rejection_region: str = "doubletail",
    big_count: int = 900,
    prior_count: float = 0.125,
) -> DGEExact:
    """Two-group negative-binomial exact test compatible with edgeR columns."""

    if not isinstance(object, DGEList):
        raise TypeError("exact_test expects a DGEList.")
    groups = _get_group(object, None)
    levels = _group_levels(object.samples["group"] if "group" in object.samples.columns else groups)
    if pair is None:
        pair = levels[:2]
    if len(pair) != 2:
        raise ValueError("pair must have length 2.")
    pair = [levels[p - 1] if isinstance(p, int) else p for p in pair]
    mask = np.isin(groups, pair)
    y = object.counts.loc[:, mask].to_numpy(dtype=float)
    sub_groups = groups[mask]
    lib = effective_lib_sizes(object)[mask]
    disp = _resolve_dispersion(object, dispersion)
    if np.ndim(disp) == 0:
        disp = np.full(object.counts.shape[0], float(disp))
    disp = np.asarray(disp, dtype=float)

    j1 = sub_groups == pair[0]
    j2 = sub_groups == pair[1]
    raw_y1 = y[:, j1]
    raw_y2 = y[:, j2]
    pseudo = _equalize_for_exact_test(y, lib, disp)
    y1 = pseudo[:, j1]
    y2 = pseudo[:, j2]
    n1 = y1.shape[1]
    n2 = y2.shape[1]

    # edgeR logFC uses NB one-group fitted abundance with a small prior count.
    scaled_prior = prior_count * lib / np.mean(lib)
    offset_aug = np.log(lib + 2.0 * scaled_prior)
    abundance1 = _one_group_abundance(raw_y1 + scaled_prior[j1][None, :], offset_aug[j1], disp)
    abundance2 = _one_group_abundance(raw_y2 + scaled_prior[j2][None, :], offset_aug[j2], disp)
    logfc = (abundance2 - abundance1) / np.log(2.0)

    rejection_region = rejection_region.lower()
    if rejection_region == "doubletail":
        pvals = _exact_test_double_tail(y1, y2, disp, big_count=big_count)
    elif rejection_region == "smallp":
        pvals = _exact_test_by_smallp(y1, y2, disp, big_count=big_count)
    elif rejection_region == "deviance":
        pvals = _exact_test_by_deviance(y1, y2, disp, big_count=big_count)
    else:
        raise ValueError("rejection_region must be one of doubletail, smallp, deviance.")
    table = pd.DataFrame(
        {
            "logFC": logfc,
            "logCPM": object.AveLogCPM if object.AveLogCPM is not None else ave_log_cpm(object),
            "PValue": pvals,
        },
        index=object.counts.index,
    )
    return DGEExact(table=table, comparison=(str(pair[0]), str(pair[1])), genes=object.genes)


def top_tags(x: DGEExact, n: int | float = 10, adjust_method: str = "BH", sort_by: str = "PValue") -> pd.DataFrame:
    tab = x.table.copy()
    tab["FDR"] = _p_adjust_bh(tab["PValue"].to_numpy()) if adjust_method.upper() == "BH" else tab["PValue"]
    if sort_by.lower() != "none":
        tab = tab.sort_values(sort_by, kind="stable")
    if np.isfinite(n):
        tab = tab.head(int(n))
    return tab


def decide_tests_dge(object, adjust_method: str = "BH", p_value: float = 0.05, lfc: float = 0.0) -> TestResults:
    """edgeR-style DE calls for DGEExact/DGELRT results."""

    if not isinstance(object, (DGEExact, DGELRT)):
        raise TypeError("decide_tests_dge expects a DGEExact or DGELRT object.")
    tab = object.table
    if "PValue" not in tab.columns:
        raise ValueError("result table must contain a PValue column.")
    adj = _p_adjust(tab["PValue"].to_numpy(dtype=float), adjust_method)
    calls = (adj < p_value).astype(int)
    if "logFC" in tab.columns:
        logfc = tab["logFC"].to_numpy(dtype=float)
        calls[(calls == 1) & (logfc < 0)] = -1
        calls[np.abs(logfc) < lfc] = 0
        levels = (-1, 0, 1)
        labels = ("Down", "NotSig", "Up")
    else:
        logfc_cols = [c for c in tab.columns if str(c).startswith("logFC")]
        if lfc > 0 and logfc_cols:
            small = (np.abs(tab[logfc_cols].to_numpy(dtype=float)) >= lfc).sum(axis=1) == 0
            calls[small] = 0
        levels = (0, 1)
        labels = ("NotSig", "Sig")
    comparison = object.comparison
    col = "-".join(reversed(comparison)) if isinstance(comparison, tuple) else str(comparison)
    frame = pd.DataFrame(calls.astype(int), index=tab.index, columns=[col])
    return TestResults(frame, levels=levels, labels=labels)


def _default_design_from_group(x: DGEList) -> pd.DataFrame:
    if "group" not in x.samples.columns:
        return pd.DataFrame({"Intercept": np.ones(x.counts.shape[1])}, index=x.counts.columns)
    group = pd.Categorical(x.samples["group"])
    if len(group.categories) <= 1:
        return pd.DataFrame({"Intercept": np.ones(x.counts.shape[1])}, index=x.counts.columns)
    data = {"Intercept": np.ones(x.counts.shape[1])}
    base = group.categories[0]
    for lev in group.categories[1:]:
        data[str(lev)] = (group == lev).astype(float)
    return pd.DataFrame(data, index=x.counts.columns)


def _fit_nb_glm_row(y: np.ndarray, design: np.ndarray, offset: np.ndarray, dispersion: float, weights=None):
    y = np.asarray(y, dtype=float)
    X = np.asarray(design, dtype=float)
    off = np.asarray(offset, dtype=float)
    w = np.ones_like(y, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    start = np.zeros(X.shape[1], dtype=float)
    with np.errstate(divide="ignore"):
        start[0] = np.log(max(np.sum(y), 1e-8) / np.sum(np.exp(off)))

    def objective(beta):
        eta = off + X @ beta
        mu = np.exp(np.clip(eta, -745, 700))
        return -_nbinom_loglik_row(y, mu, dispersion, weights=w)

    def gradient(beta):
        eta = off + X @ beta
        mu = np.exp(np.clip(eta, -745, 700))
        if dispersion <= 0:
            score_obs = y - mu
        else:
            score_obs = (y - mu) / (1.0 + dispersion * mu)
        score_obs = score_obs * w
        return -(X.T @ score_obs)

    res = optimize.minimize(
        objective,
        start,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 200},
    )
    beta = res.x if np.all(np.isfinite(res.x)) else start
    mu = np.exp(np.clip(off + X @ beta, -745, 700))
    return beta, mu


def _estimate_disp_prior_df(
    object: DGEList,
    *,
    sel: np.ndarray,
    design: np.ndarray | None,
    groups: np.ndarray,
    lib: np.ndarray,
    dispersion_trend,
    covariate: np.ndarray | None,
    robust: bool,
    winsor_tail_p: tuple[float, float],
):
    counts_sel = object.counts.iloc[np.flatnonzero(sel)]
    if design is None:
        tmp = DGEList(counts_sel, lib_size=lib, norm_factors=np.ones_like(lib), group=groups)
        design_for_fit = _default_design_from_group(tmp)
    else:
        design_for_fit = design
    fit = glm_fit(
        counts_sel,
        design=design_for_fit,
        dispersion=dispersion_trend,
        offset=np.log(lib),
        prior_count=0,
        weights=None if object.weights is None else object.weights.iloc[np.flatnonzero(sel)],
    )
    from . import _glm_batch as gb
    zerofit = (fit.counts.to_numpy(dtype=float) < 1e-4) & (fit.fitted_values < 1e-4)
    df = gb.resid_df(zerofit, np.asarray(design_for_fit, dtype=float))   # edgeR .residDF
    s2 = np.zeros_like(fit.deviance, dtype=float)
    ok = df > 0
    s2[ok] = np.maximum(fit.deviance[ok] / df[ok], 0.0)
    df_prior, _, _ = _squeeze_var(
        s2,
        df,
        covariate=covariate,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    if np.asarray(df_prior).ndim == 0:
        return float(df_prior)
    out = np.full(object.counts.shape[0], np.inf, dtype=float)
    out[sel] = np.asarray(df_prior, dtype=float)
    return out


def _squeeze_var(
    var: np.ndarray,
    df: np.ndarray,
    covariate=None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
):
    var = np.asarray(var, dtype=float).copy()
    df = np.asarray(df, dtype=float)
    if robust:
        dfp = df[np.isfinite(df) & (df > 0)]
        if covariate is None and dfp.size and np.nanmin(dfp) == np.nanmax(dfp):
            return _squeeze_var_robust_equal_df(var, df, winsor_tail_p=winsor_tail_p)
    dfp = df[df > 0]
    if not robust or (dfp.size and np.min(dfp) != np.max(dfp)):
        # limma 3.68: unequal residual df use fitFDistUnequalDF1 (robust included)
        from . import _limma
        return _limma.squeeze_var(var, df, covariate=covariate, robust=robust)
    n = var.size
    if n < 3:
        return 0.0, var.copy(), var.copy()
    if df.size > 1:
        var[df == 0] = 0.0
    ok = np.isfinite(df) & (df > 1e-15) & np.isfinite(var) & (var > -1e-15)
    if not np.any(ok):
        return np.nan, np.nan, np.full_like(var, np.nan)
    x = np.maximum(var[ok], 0.0)
    med = np.median(x)
    if med == 0:
        med = 1.0
    x = np.maximum(x, 1e-5 * med)
    df_ok = df[ok]
    z = np.log(x)
    e = z + _logmdigamma(df_ok / 2.0)
    emean = float(np.mean(e))
    evar = float(np.sum((e - emean) ** 2) / max(e.size - 1, 1))
    evar = evar - float(np.mean(special.polygamma(1, df_ok / 2.0)))
    if evar > 0:
        df_prior = 2.0 * _trigamma_inverse(evar)
        var_prior = float(np.exp(emean - _logmdigamma(df_prior / 2.0)))
    else:
        df_prior = float("inf")
        var_prior = float(np.mean(x))
    if np.isfinite(df_prior):
        var_post = (df * var + df_prior * var_prior) / (df + df_prior)
    else:
        var_post = np.full_like(var, var_prior)
    return df_prior, var_prior, var_post


def _squeeze_var_robust_equal_df(
    var: np.ndarray,
    df: np.ndarray,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
):
    df_prior, var_prior, var_post = _squeeze_var(var, df, robust=False)
    good = np.isfinite(var) & np.isfinite(df) & (df > 1e-15) & (var > 0)
    if np.sum(good) < 3 or not np.isfinite(df_prior):
        return df_prior, var_prior, var_post

    z = np.log(np.maximum(var[good] / max(float(var_prior), 1e-300), 1e-300))
    lo, hi = np.quantile(z, [winsor_tail_p[0], 1.0 - winsor_tail_p[1]], method="linear")
    zw = np.clip(z, lo, hi)
    zw_mean = float(np.mean(zw))
    zw_var = float(np.var(zw, ddof=1))
    base_var = float(np.var(z, ddof=1))
    if zw_var <= 0 or base_var <= 0:
        return df_prior, var_prior, var_post

    # Approximate limma/edgeR's robust df shrinkage: keep the central
    # winsorized moment estimate as the shared prior df, but reduce prior df
    # for genes whose upper-tail F probability is smaller than its empirical
    # tail probability.
    robust_df = max(float(df_prior) * base_var / zw_var, float(df_prior))
    df1 = float(np.nanmax(df[good]))
    fstat = np.maximum(var[good] / max(float(var_prior), 1e-300), 1e-300)
    log_tail = stats.f.logsf(fstat, df1, robust_df)
    empirical = np.log(np.sum(good) - rankdata(fstat, method="average") + 0.5) - np.log(np.sum(good))
    prob_not_outlier = np.exp(np.minimum(log_tail - empirical, 0.0))
    df_prior_good = prob_not_outlier * robust_df
    order = np.argsort(log_tail, kind="mergesort")
    ordered = df_prior_good[order]
    running = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    imin = int(np.argmin(running))
    ordered[: imin + 1] = running[imin]
    df_prior_good[order] = np.maximum.accumulate(ordered)

    df_prior_vec = np.full_like(var, robust_df, dtype=float)
    df_prior_vec[good] = df_prior_good
    var_post_vec = (df * var + df_prior_vec * float(var_prior)) / (df + df_prior_vec)
    return df_prior_vec, float(var_prior), var_post_vec


def _logmdigamma(x):
    return np.log(x) - special.digamma(x)


def _trigamma_inverse(x: float, max_iter: int = 50, atol: float = 1e-8) -> float:
    if x <= 0:
        return float("inf")
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(max_iter):
        tri = special.polygamma(1, y)
        step = tri * (1.0 - tri / x) / special.polygamma(2, y)
        y += step
        if abs(step / y) < atol:
            break
    return float(y)


def _nbinom_loglik_row(y: np.ndarray, mu: np.ndarray, dispersion: float, weights=None) -> float:
    mu = np.maximum(mu, 1e-300)
    w = 1.0 if weights is None else np.asarray(weights, dtype=float)
    if dispersion <= 0:
        return float(np.sum(w * stats.poisson.logpmf(y, mu)))
    r = 1.0 / dispersion
    return float(
        np.sum(
            w
            * (
                special.gammaln(y + r)
                - special.gammaln(r)
                - special.gammaln(y + 1.0)
                + r * (np.log(r) - np.log(r + mu))
                + y * (np.log(mu) - np.log(r + mu))
            )
        )
    )


def _nbinom_deviance_row(y: np.ndarray, mu: np.ndarray, dispersion: float, weights=None) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    w = 1.0 if weights is None else np.asarray(weights, dtype=float)
    if dispersion <= 0:
        term = np.zeros_like(y, dtype=float)
        nz = y > 0
        term[nz] = y[nz] * np.log(y[nz] / mu[nz])
        return float(2.0 * np.sum(w * (term - (y - mu))))
    phi = dispersion
    term1 = np.zeros_like(y, dtype=float)
    nz = y > 0
    term1[nz] = y[nz] * np.log(y[nz] / mu[nz])
    term2 = (y + 1.0 / phi) * np.log((1.0 + phi * y) / (1.0 + phi * mu))
    return float(2.0 * np.sum(w * (term1 - term2)))


def _nbinom_unit_deviance(y, mu, dispersion):
    y = np.asarray(y, dtype=float) + 1e-8
    mu = np.asarray(mu, dtype=float) + 1e-8
    phi = np.asarray(dispersion, dtype=float)
    out = np.empty(np.broadcast_shapes(y.shape, mu.shape, phi.shape), dtype=float)
    yy, mm, pp = np.broadcast_arrays(y, mu, phi)
    small = pp < 1e-4
    if np.any(small):
        resid = yy[small] - mm[small]
        out[small] = 2.0 * (
            yy[small] * np.log(yy[small] / mm[small])
            - resid
            - 0.5 * resid * resid * pp[small] * (1.0 + pp[small] * (2.0 / 3.0 * resid - yy[small]))
        )
    large = (~small) & (mm * pp > 1e6)
    if np.any(large):
        out[large] = 2.0 * ((yy[large] - mm[large]) / mm[large] - np.log(yy[large] / mm[large])) * mm[large] / (1.0 + mm[large] * pp[large])
    mid = ~(small | large)
    if np.any(mid):
        invphi = 1.0 / pp[mid]
        out[mid] = 2.0 * (
            yy[mid] * np.log(yy[mid] / mm[mid])
            + (yy[mid] + invphi) * np.log((mm[mid] + invphi) / (yy[mid] + invphi))
        )
    return np.maximum(out, 0.0)


def _get_group(x: DGEList, group):
    if group is not None:
        return np.asarray(group)
    if "group" not in x.samples.columns:
        return np.ones(x.counts.shape[1])         # R DGEList default: a single group
    return x.samples["group"].to_numpy()


def _group_levels(group) -> list:
    """Levels in R factor order: categorical order if given, else sorted unique values."""
    g = pd.Series(group)
    if isinstance(g.dtype, pd.CategoricalDtype):
        present = set(g.dropna())
        return [c for c in g.cat.categories if c in present]
    return sorted(pd.unique(g.dropna()))


def _resolve_dispersion(x: DGEList, dispersion):
    if dispersion is None or dispersion == "auto":
        if x.tagwise_dispersion is not None:
            return x.tagwise_dispersion
        if x.trended_dispersion is not None:
            return x.trended_dispersion
        if x.common_dispersion is not None:
            return x.common_dispersion
        raise ValueError("No dispersion found. Run estimate_disp or provide dispersion.")
    if dispersion == "common":
        return x.common_dispersion
    if dispersion == "tagwise":
        return x.tagwise_dispersion
    if dispersion == "trended":
        return x.trended_dispersion
    return dispersion


def _one_group_abundance(y: np.ndarray, offset: np.ndarray, dispersion: np.ndarray) -> np.ndarray:
    """edgeR mglmOneGroup (Fisher scoring, maxit=50, tol=1e-10)."""
    from . import _glm_batch as gb
    y = np.asarray(y, dtype=float)
    off = np.broadcast_to(np.asarray(offset, dtype=float), y.shape)
    disp = np.asarray(dispersion, dtype=float)
    phi = np.broadcast_to(disp.reshape(-1, 1) if disp.ndim == 1 else disp, y.shape)
    return gb.mglm_one_group(y, off, phi, np.ones_like(y), maxit=50, tol=1e-10)


def _equalize_for_exact_test(y: np.ndarray, lib_size: np.ndarray, dispersion: np.ndarray) -> np.ndarray:
    lib_average = float(np.exp(np.mean(np.log(lib_size))))
    abundance = _one_group_abundance(y, np.log(lib_size), dispersion)
    expr = np.exp(abundance)
    input_mean = expr[:, None] * lib_size[None, :]
    output_mean = np.broadcast_to(expr[:, None] * lib_average, y.shape).copy()
    return q2qnbinom(y, input_mean=input_mean, output_mean=output_mean, dispersion=dispersion)


def _split_into_groups(y: np.ndarray, group: np.ndarray) -> list[np.ndarray]:
    group = np.asarray(group)
    return [y[:, group == lev] for lev in pd.unique(group)]


def _cond_loglik_delta(y: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0 or delta >= 1:
        return np.full(y.shape[0], -np.inf)
    r = 1.0 / delta - 1.0
    n = y.shape[1]
    m = np.mean(y, axis=1)
    return (
        np.sum(special.gammaln(y + r), axis=1)
        + special.gammaln(n * r)
        - special.gammaln(n * (m + r))
        - n * special.gammaln(r)
    )


def _common_cond_loglik_delta(groups: list[np.ndarray], delta: float) -> float:
    total = 0.0
    for y in groups:
        total += float(np.sum(_cond_loglik_delta(y, delta)))
    return total


def _classic_cond_loglik_grid(
    y: np.ndarray,
    *,
    groups: np.ndarray,
    lib_size: np.ndarray,
    equalize_dispersion: float,
    sel: np.ndarray,
    grid_length: int,
    grid_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(grid_range[0], grid_range[1], grid_length)
    spline_disp = 0.1 * 2.0**theta
    grid_delta = spline_disp / (1.0 + spline_disp)
    pseudo, _ = equalize_lib_sizes(y, group=groups, dispersion=equalize_dispersion, lib_size=lib_size)
    split = _split_into_groups(pseudo[sel], groups)
    l0 = np.zeros((int(np.sum(sel)), grid_length), dtype=float)
    for j, delta in enumerate(grid_delta):
        for group_y in split:
            l0[:, j] += _cond_loglik_delta(group_y, float(delta))
    return theta, l0


def _design_adjusted_profile_loglik_grid(
    y: np.ndarray,
    *,
    design: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None = None,
    grid_length: int,
    grid_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """estimateDisp.default design branch: Cox-Reid APL on the dispersion grid.

    As in edgeR, observations fitted as exactly zero (at dispersion 0.05) are
    dropped per zero-pattern group and the design is reduced to an estimable
    set of columns; fits along the grid are warm-started from the previous
    grid point.
    """
    from . import _glm_batch as gb
    theta = np.linspace(grid_range[0], grid_range[1], grid_length)
    spline_disp = 0.1 * 2.0**theta
    y = np.asarray(y, dtype=float)
    off = np.asarray(offset, dtype=float)
    offset_mat = np.broadcast_to(off, y.shape).astype(float) if off.ndim == 1 else off.astype(float)
    W = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    l0 = np.zeros((y.shape[0], grid_length), dtype=float)
    _, fitted0, _ = _fit_glm_batch(y, design, offset_mat, np.full_like(y, 0.05), W)
    zerofit = (y < 1e-4) & (fitted0 < 1e-4)
    for subg in gb.combo_groups(zerofit):
        nz = ~zerofit[subg[0]]
        if not nz.any():
            continue
        if nz.all():
            redesign = design
        else:
            redesign = design[nz]
            redesign = redesign[:, gb.qr_rank_columns(redesign)]
            if redesign.shape[0] == redesign.shape[1]:
                continue
        cy, co, cw = y[np.ix_(subg, nz)], offset_mat[np.ix_(subg, nz)], W[np.ix_(subg, nz)]
        last = None
        for j, disp in enumerate(spline_disp):
            phi = np.full_like(cy, disp)
            last, mu, _ = _fit_glm_batch(cy, redesign, co, phi, cw, start=last)
            l0[subg, j] = gb.compute_apl(cy, mu, phi, cw, redesign)
    return theta, l0


def _adjusted_profile_loglik_row(y: np.ndarray, mu: np.ndarray, design: np.ndarray, dispersion: float, weights=None) -> float:
    ll = _nbinom_loglik_row(y, mu, dispersion, weights=weights)
    if dispersion <= 0:
        working = mu
    else:
        working = mu / (1.0 + dispersion * mu)
    if weights is not None:
        working = working * np.asarray(weights, dtype=float)
    xtwx = design.T @ (working[:, None] * design)
    sign, logdet = np.linalg.slogdet(xtwx)
    if sign <= 0 or not np.isfinite(logdet):
        vals = np.linalg.eigvalsh(xtwx)
        vals = np.maximum(vals, 1e-10)
        logdet = float(np.sum(np.log(vals)))
    return float(ll - 0.5 * logdet)


def _maximize_interpolant(theta: np.ndarray, loglik: np.ndarray) -> np.ndarray:
    """edgeR maximizeInterpolant: maximum of the FMM cubic spline through each row."""
    from . import _glm_batch as gb
    y = np.atleast_2d(np.asarray(loglik, dtype=float))
    out = np.full(y.shape[0], np.nan)
    ok = np.isfinite(y).all(axis=1)
    if ok.any():
        out[ok] = gb.maximize_interpolant(theta, y[ok])
    return out


def _choose_lowess_span(
    n: int,
    *,
    small_n: int = 50,
    min_span: float = 0.3,
    power: float = 1.0 / 3.0,
) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")
    return float(min(min_span + (1.0 - min_span) * (small_n / n) ** power, 1.0))


def _moving_average_by_col(x: np.ndarray, width: int = 5, full_length: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    width = int(width)
    if width <= 1:
        return x.copy()
    n, m = x.shape
    if width > n:
        width = n
    if full_length:
        half1 = int(np.ceil(width / 2.0))
        half2 = int(np.floor(width / 2.0))
        padded = np.vstack([np.zeros((half1, m)), x, np.zeros((half2, m))])
    else:
        if width == n:
            return np.mean(x, axis=0, keepdims=True)
        padded = np.vstack([np.zeros((1, m)), x])
    cs = np.cumsum(padded, axis=0)
    smoothed = cs[width:] - cs[:-width]
    n3 = smoothed.shape[0]
    denom = np.full(n3, width, dtype=float)
    if full_length:
        if half1 > 1:
            denom[: half1 - 1] = width - np.arange(half1 - 1, 0, -1)
        if half2 > 0:
            denom[n3 - half2 :] = width - np.arange(1, half2 + 1)
    return smoothed / denom[:, None]


def _moving_average_loglik(loglik: np.ndarray, covariate: np.ndarray, span: float) -> np.ndarray:
    n = loglik.shape[0]
    width = int(np.floor(span * n))
    order = np.argsort(covariate, kind="mergesort")
    reverse = np.empty_like(order)
    reverse[order] = np.arange(order.size)
    return _moving_average_by_col(loglik[order], width=width, full_length=True)[reverse]


def _loess_by_col(y: np.ndarray, x: np.ndarray | None = None, span: float = 0.5) -> np.ndarray:
    """Degree-0 tricube loess smoother matching edgeR's ``loessByCol`` role."""

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n = y.shape[0]
    if x is None:
        x = np.arange(1, n + 1, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    if x.size != n:
        raise ValueError("x length must match y rows.")
    nspan = min(int(np.floor(span * n)), n)
    if nspan <= 1:
        return y.copy()
    fitted = np.empty_like(y, dtype=float)
    for i in range(n):
        dist = np.abs(x - x[i])
        order = np.argsort(dist, kind="mergesort")
        keep = order[:nspan]
        dmax = float(dist[keep[-1]])
        if dmax <= 0:
            w = np.ones(keep.size, dtype=float)
        else:
            u = dist[keep] / dmax
            w = (1.0 - u**3) ** 3
            w[u >= 1.0] = 0.0
            if not np.any(w > 0):
                w = np.ones(keep.size, dtype=float)
        fitted[i] = np.sum(y[keep] * w[:, None], axis=0) / np.sum(w)
    return fitted


def _locfit_by_col(
    y: np.ndarray,
    x: np.ndarray | None = None,
    span: float = 0.5,
    degree: int = 0,
    weights=None,
) -> np.ndarray:
    """edgeR locfitByCol, reproducing the R locfit package (tree evaluation + interpolation)."""
    from ._locfit import locfit_by_col
    return locfit_by_col(y, x=x, weights=weights, span=span, degree=degree)


def _average_ql_dispersion(s2: np.ndarray, df: np.ndarray, ave_log_cpm: np.ndarray) -> float:
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 1e-8)
    if not np.any(ok):
        return 1.0
    y = np.sqrt(np.sqrt(s2[ok]))
    x = np.asarray(ave_log_cpm, dtype=float)[ok]
    # edgeR uses lowess f=0.5, iter=3 and the 90% quantile of the trend,
    # bounded below by 1 on the QL dispersion scale.
    line = lowess(y, x, frac=0.5, it=3, delta=0.01 * (np.max(x) - np.min(x)), return_sorted=False)
    q = float(np.quantile(line, 0.9, method="linear"))
    return max(q**4, 1.0)


def q2qnbinom(x, input_mean, output_mean, dispersion=0):
    """Approximate edgeR ``q2qnbinom`` mapping between NB distributions."""

    x = np.asarray(x, dtype=float)
    input_mean = np.asarray(input_mean, dtype=float).copy()
    output_mean = np.asarray(output_mean, dtype=float).copy()
    if np.any(x < 0) or np.any(input_mean < 0) or np.any(output_mean < 0):
        raise ValueError("x, input_mean and output_mean must be non-negative.")
    disp = np.asarray(dispersion, dtype=float)
    if disp.ndim == 0:
        disp = np.full(x.shape[0], float(disp))
    disp = disp[:, None]
    eps = 1e-14
    zero = (input_mean < eps) | (output_mean < eps)
    input_mean[zero] += 0.25
    output_mean[zero] += 0.25

    ri = 1.0 + disp * input_mean
    vi = input_mean * ri
    ro = 1.0 + disp * output_mean
    vo = output_mean * ro
    above = x >= input_mean
    out = np.empty_like(x, dtype=float)

    p1 = np.empty_like(x, dtype=float)
    p2 = np.empty_like(x, dtype=float)
    q1 = np.empty_like(x, dtype=float)
    q2 = np.empty_like(x, dtype=float)

    sdi, sdo = np.sqrt(vi), np.sqrt(vo)
    if np.any(above):
        a = above
        p1[a] = special.log_ndtr(-(x[a] - input_mean[a]) / sdi[a])                       # pnorm(upper, log)
        p2[a] = np.log(special.gammaincc(input_mean[a] / ri[a], x[a] / ri[a]))           # pgamma(upper, log)
        q1[a] = output_mean[a] - sdo[a] * special.ndtri_exp(p1[a])                       # qnorm(upper, log)
        q2[a] = special.gammainccinv(output_mean[a] / ro[a], np.exp(p2[a])) * ro[a]      # qgamma(upper, log)
    below = ~above
    if np.any(below):
        b = below
        p1[b] = special.log_ndtr((x[b] - input_mean[b]) / sdi[b])
        p2[b] = np.log(special.gammainc(input_mean[b] / ri[b], x[b] / ri[b]))
        q1[b] = output_mean[b] + sdo[b] * special.ndtri_exp(p1[b])
        q2[b] = special.gammaincinv(output_mean[b] / ro[b], np.exp(p2[b])) * ro[b]
    out = (q1 + q2) / 2.0
    return out


def _exact_test_double_tail(y1: np.ndarray, y2: np.ndarray, dispersion: np.ndarray, big_count: int = 900) -> np.ndarray:
    raw_s1 = y1.sum(axis=1)
    raw_s2 = y2.sum(axis=1)
    s1 = np.rint(raw_s1).astype(int)
    s2 = np.rint(raw_s2).astype(int)
    n1 = y1.shape[1]
    n2 = y2.shape[1]
    total = s1 + s2
    pvals = np.ones(y1.shape[0], dtype=float)
    for i, (a, b, s, disp) in enumerate(zip(s1, s2, total, dispersion)):
        if s == 0:
            pvals[i] = 1.0
            continue
        if disp <= 0:
            pvals[i] = stats.binomtest(a, n=s, p=n1 / (n1 + n2)).pvalue
            continue
        if a > big_count and b > big_count:
            pvals[i] = _exact_test_beta_approx_row(raw_s1[i], raw_s2[i], n1, n2, disp)
            continue
        mu = s / (n1 + n2)
        mu1 = n1 * mu
        mu2 = n2 * mu
        if a == mu1:
            pvals[i] = 1.0
            continue
        size1 = n1 / disp
        size2 = n2 / disp
        size_total = (n1 + n2) / disp
        if a < mu1:
            xs = np.arange(0, a + 1)
        else:
            xs = np.arange(a, s + 1)
        p_top = _nbinom_pmf_mu(xs, size1, mu1) * _nbinom_pmf_mu(s - xs, size2, mu2)
        p_bot = _nbinom_pmf_mu(s, size_total, s)
        pvals[i] = min(1.0, 2.0 * np.sum(p_top) / p_bot)
    return pvals


def _exact_test_beta_approx_row(s1: float, s2: float, n1: int, n2: int, dispersion: float) -> float:
    total = s1 + s2
    if total <= 0:
        return 1.0
    mu = total / (n1 + n2)
    alpha1 = n1 * mu / (1.0 + dispersion * mu)
    alpha2 = n2 / n1 * alpha1
    med = stats.beta.ppf(0.5, alpha1, alpha2)
    pval = 1.0
    if (s1 + 0.5) / total < med:
        pval = 2.0 * stats.beta.cdf((s1 + 0.5) / total, alpha1, alpha2)
    elif (s1 - 0.5) / total > med:
        pval = 2.0 * stats.beta.sf((s1 - 0.5) / total, alpha1, alpha2)
    return min(float(pval), 1.0)


def _exact_test_by_smallp(y1: np.ndarray, y2: np.ndarray, dispersion: np.ndarray, big_count: int = 900) -> np.ndarray:
    n1 = y1.shape[1]
    n2 = y2.shape[1]
    if n1 == n2:
        return _exact_test_double_tail(y1, y2, dispersion, big_count=big_count)
    s1 = np.rint(y1.sum(axis=1)).astype(int)
    s2 = np.rint(y2.sum(axis=1)).astype(int)
    total = s1 + s2
    pvals = np.ones(y1.shape[0], dtype=float)
    if np.all(dispersion <= 0):
        return np.array([stats.binomtest(a, n=s, p=n1 / (n1 + n2)).pvalue if s > 0 else 1.0 for a, s in zip(s1, total)])
    if np.any(dispersion <= 0):
        raise ValueError("dispersion must be either all zero or all positive.")
    for i, (a, b, s, disp) in enumerate(zip(s1, s2, total, dispersion)):
        if s == 0:
            continue
        mu = s / (n1 + n2)
        size1 = n1 / disp
        size2 = n2 / disp
        xs = np.arange(0, s + 1)
        p_top = _nbinom_pmf_mu(xs, size1, n1 * mu) * _nbinom_pmf_mu(s - xs, size2, n2 * mu)
        p_obs = _nbinom_pmf_mu(a, size1, n1 * mu) * _nbinom_pmf_mu(b, size2, n2 * mu)
        p_bot = _nbinom_pmf_mu(s, size1 + size2, s)
        pvals[i] = min(1.0, float(np.sum(p_top[p_top <= p_obs]) / p_bot))
    # edgeR's R implementation returns min(pvals, 1), which is recycled into
    # the exactTest table. Preserve that behavior for parity.
    return np.full_like(pvals, min(float(np.min(pvals)), 1.0))


def _exact_test_by_deviance(y1: np.ndarray, y2: np.ndarray, dispersion: np.ndarray, big_count: int = 900) -> np.ndarray:
    n1 = y1.shape[1]
    n2 = y2.shape[1]
    if n1 == n2:
        return _exact_test_double_tail(y1, y2, dispersion, big_count=big_count)
    s1 = np.rint(y1.sum(axis=1)).astype(int)
    s2 = np.rint(y2.sum(axis=1)).astype(int)
    total = s1 + s2
    pvals = np.ones(y1.shape[0], dtype=float)
    if np.all(dispersion <= 0):
        return np.array([stats.binomtest(a, n=s, p=n1 / (n1 + n2)).pvalue if s > 0 else 1.0 for a, s in zip(s1, total)])
    if np.any(dispersion <= 0):
        raise ValueError("dispersion must be either all zero or all positive.")
    for i, (a, b, s, disp) in enumerate(zip(s1, s2, total, dispersion)):
        if s == 0:
            continue
        mu = s / (n1 + n2)
        mu1 = n1 * mu
        mu2 = n2 * mu
        size1 = n1 / disp
        size2 = n2 / disp
        phi1 = 1.0 / size1
        phi2 = 1.0 / size2
        p = size1 / (size1 + mu1)
        obsdev = _nbinom_unit_deviance(a, mu1, phi1) + _nbinom_unit_deviance(b, mu2, phi2)
        prob = 0.0
        j = 0
        while j <= s:
            dev = _nbinom_unit_deviance(j, mu1, phi1) + _nbinom_unit_deviance(s - j, mu2, phi2)
            if obsdev <= dev:
                prob += stats.nbinom.pmf(j, size1, p) * stats.nbinom.pmf(s - j, size2, p)
            else:
                break
            j += 1
        for k in range(0, s - j + 1):
            dev = _nbinom_unit_deviance(k, mu2, phi2) + _nbinom_unit_deviance(s - k, mu1, phi1)
            if obsdev <= dev:
                prob += stats.nbinom.pmf(k, size2, p) * stats.nbinom.pmf(s - k, size1, p)
            else:
                break
        p_bot = _nbinom_pmf_mu(s, size1 + size2, s)
        pvals[i] = min(1.0, float(prob / p_bot))
    return pvals


def _nbinom_pmf_mu(k, size, mu):
    p = size / (size + mu)
    return stats.nbinom.pmf(k, size, p)


def _p_adjust_bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    order = np.argsort(p[ok])
    sorted_p = p[ok][order]
    n = sorted_p.size
    adj = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    idx = np.flatnonzero(ok)
    out[idx[order]] = adj
    return out


def _p_adjust(p: np.ndarray, method: str = "BH") -> np.ndarray:
    method_l = method.lower()
    p = np.asarray(p, dtype=float)
    if method_l in {"bh", "fdr"}:
        return _p_adjust_bh(p)
    if method_l in {"none", "raw"}:
        return p.copy()
    if method_l == "bonferroni":
        return np.minimum(p * np.sum(np.isfinite(p)), 1.0)
    raise NotImplementedError("Only BH/fdr, bonferroni and none p-value adjustments are implemented.")
