"""pyedger run on every benchmark task, mirroring 02_run_edger.R step by step.

Usage: python 03_run_pyedger.py <tasks_dir> <out_dir> <n_timing_reps> [task_glob]
The engine is chosen by the environment: PYEDGER_NO_NUMBA=1 forces the NumPy
path; otherwise the numba kernels run with NUMBA_NUM_THREADS threads.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

import pyedger


def run_pipeline(X: pd.DataFrame, group: pd.Categorical) -> dict:
    tm = {}
    t0 = time.perf_counter()
    y = pyedger.DGEList(X, group=group)
    keep = np.asarray(pyedger.filterByExpr(y, group=np.asarray(group)))
    # R: y[keep, , keep.lib.sizes = FALSE]
    y = pyedger.calcNormFactors(pyedger.DGEList(X.loc[keep], group=group))
    design = np.column_stack([np.ones(len(group)), (np.asarray(group) == "Tumor").astype(float)])
    tm["filter_norm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    y = pyedger.estimateDisp(y, design=design)
    tm["estimateDisp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    qfit = pyedger.glmQLFit(y, design)
    ql = pyedger.glmQLFTest(qfit, coef=1)
    tm["QL"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    gfit = pyedger.glmFit(y, design)
    lrt = pyedger.glmLRT(gfit, coef=1)
    tm["LRT"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    et = pyedger.exactTest(y)
    tm["exact"] = time.perf_counter() - t0
    return dict(y=y, keep=keep, qfit=qfit, ql=ql, lrt=lrt, et=et, time=tm)


def bh(p):
    from scipy.stats import false_discovery_control
    return false_discovery_control(np.asarray(p, dtype=float), method="bh")


def engine() -> str:
    if os.environ.get("PYEDGER_NO_NUMBA"):
        return "numpy"
    return f"numba{os.environ.get('NUMBA_NUM_THREADS', 'all')}"


def main():
    tasks_dir, out_dir, reps = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    pattern = sys.argv[4] if len(sys.argv) > 4 else "*"
    out_dir.mkdir(parents=True, exist_ok=True)
    # warm-up on the smallest task so numba compilation / cache loading is not timed
    warm = sorted(tasks_dir.glob("sub03_*"))[0]
    Xw = pd.read_csv(warm / "counts.tsv.gz", sep="\t", index_col=0)
    gw = pd.Categorical(pd.read_csv(warm / "samples.tsv", sep="\t", index_col=0).loc[Xw.columns, "group"], categories=["Normal", "Tumor"])
    t0 = time.perf_counter()
    run_pipeline(Xw, gw)
    (out_dir / "warmup.json").write_text(json.dumps({"engine": engine(), "seconds": time.perf_counter() - t0}))
    for td in sorted(tasks_dir.glob(pattern)):
        if not (td / "counts.tsv.gz").exists():
            continue
        od = out_dir / td.name
        if (od / "timing.json").exists():
            continue
        od.mkdir(exist_ok=True)
        X = pd.read_csv(td / "counts.tsv.gz", sep="\t", index_col=0)
        s = pd.read_csv(td / "samples.tsv", sep="\t", index_col=0)
        group = pd.Categorical(s.loc[X.columns, "group"], categories=["Normal", "Tumor"])
        times = []
        for _ in range(reps):
            res = run_pipeline(X, group)
            times.append(res["time"])
        y, qfit = res["y"], res["qfit"]
        qt, lt, et = res["ql"].table, res["lrt"].table, res["et"].table
        out = pd.DataFrame({
            "gene": y.counts.index,
            "AveLogCPM": y.AveLogCPM,
            "trended": y.trended_dispersion,
            "tagwise": y.tagwise_dispersion,
            "ql_logFC": qt["logFC"].to_numpy(),
            "ql_F": qt["F"].to_numpy(),
            "ql_PValue": qt["PValue"].to_numpy(),
            "ql_FDR": bh(qt["PValue"]),
            "ql_s2post": np.asarray(qfit.s2_post),
            "lrt_logFC": lt["logFC"].to_numpy(),
            "lrt_LR": lt["LR"].to_numpy(),
            "lrt_PValue": lt["PValue"].to_numpy(),
            "lrt_FDR": bh(lt["PValue"]),
            "et_logFC": et["logFC"].to_numpy(),
            "et_PValue": et["PValue"].to_numpy(),
            "et_FDR": bh(et["PValue"]),
        })
        out.to_csv(od / "results.tsv.gz", sep="\t", index=False, float_format="%.17g")
        tolist = lambda v: v.tolist() if isinstance(v, np.ndarray) else (None if v is None else float(v))
        scal = dict(
            n_keep=int(res["keep"].sum()), norm_factors=y.samples["norm.factors"].tolist(),
            common_dispersion=tolist(y.common_dispersion), prior_df=tolist(y.prior_df),
            ql_df_prior=tolist(qfit.df_prior), ql_s2_prior=tolist(qfit.s2_prior),
            ql_average_ql_dispersion=tolist(qfit.average_ql_dispersion),
        )
        (od / "scalars.json").write_text(json.dumps(scal))
        (od / "timing.json").write_text(json.dumps(dict(engine=engine(), version=getattr(pyedger, "__version__", None) or importlib.metadata.version("pyedger"), module=pyedger.__file__,
                                                        python=sys.version.split()[0], reps=times)))
        print(td.name, engine(), f"{sum(times[-1].values()):.2f}s", flush=True)


if __name__ == "__main__":
    main()
