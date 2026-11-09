"""Build the benchmark tasks from TCGA STAR gene counts (UCSC Xena GDC hub).

Each task is a genes x samples integer count matrix plus a two-level group
(Normal = solid tissue normal "11", Tumor = primary tumour "01"):

* ``full_<COHORT>``: every tumour and normal sample of the cohort (one aliquot
  per participant and sample type);
* ``sub<n>_LUAD_r<k>``: n tumours vs n normals drawn at random from TCGA-LUAD
  (n = 3, 5, 10; 20 draws each), the small-sample regime edgeR is mostly used in.

Xena stores log2(count + 1); the integer counts are recovered exactly by
round(2**x - 1); samples whose values are averages of replicate aliquots
(half-integers) are dropped.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/users/steorra/analysis/omicverse_dev/_port_reports_data")
SRC = ROOT / "tcga"
OUT = ROOT / "edger_tasks"
OUT.mkdir(exist_ok=True)
COHORTS = ["BRCA", "COAD", "KIRC", "LIHC", "LUAD", "LUSC", "PRAD", "THCA"]
SUB_N = [3, 5, 10]
SUB_REPS = 20


def load_cohort(name: str) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(SRC / f"TCGA-{name}.star_counts.tsv.gz", sep="\t", index_col=0)
    raw = np.exp2(X.to_numpy()) - 1.0
    counts = np.rint(raw)
    # Xena averages replicate aliquots of a sample -> half-integer "counts"; drop those samples
    integer = np.abs(raw - counts).max(axis=0) < 1e-6
    print(f"{name}: dropping {int((~integer).sum())} averaged (non-integer) samples", flush=True)
    X = pd.DataFrame(counts[:, integer].astype(np.int64), index=X.index, columns=X.columns[integer])
    X = X.loc[~X.index.str.endswith("_PAR_Y")]
    stype = X.columns.str[13:15]
    keep = np.isin(stype, ["01", "11"])
    X = X.loc[:, keep]
    # one aliquot per participant and sample type (sorted, so vial A comes first)
    X = X.loc[:, sorted(X.columns)]
    X = X.loc[:, ~X.columns.str[:15].duplicated()]
    X = X.loc[X.sum(axis=1) > 0]
    group = pd.Series(np.where(X.columns.str[13:15] == "11", "Normal", "Tumor"), index=X.columns)
    return X, group


def write_task(name: str, X: pd.DataFrame, group: pd.Series, meta: dict) -> None:
    d = OUT / name
    d.mkdir(exist_ok=True)
    X.to_csv(d / "counts.tsv.gz", sep="\t", compression={"method": "gzip", "compresslevel": 1})
    group.rename("group").to_csv(d / "samples.tsv", sep="\t", index_label="sample")
    meta.update(n_genes=int(X.shape[0]), n_samples=int(X.shape[1]),
                n_normal=int((group == "Normal").sum()), n_tumor=int((group == "Tumor").sum()))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(name, meta, flush=True)


def main():
    rng = np.random.default_rng(20261109)
    for c in COHORTS:
        X, g = load_cohort(c)
        write_task(f"full_{c}", X, g, {"cohort": c, "kind": "full"})
        if c == "LUAD":
            nor, tum = g.index[g == "Normal"], g.index[g == "Tumor"]
            for n in SUB_N:
                for r in range(SUB_REPS):
                    cols = list(rng.choice(nor, n, replace=False)) + list(rng.choice(tum, n, replace=False))
                    Xs = X[cols]
                    Xs = Xs.loc[Xs.sum(axis=1) > 0]
                    write_task(f"sub{n:02d}_LUAD_r{r:02d}", Xs, g[cols], {"cohort": c, "kind": "sub", "n": n, "rep": r})
    probemap = pd.read_csv(SRC / "gencode.v36.annotation.gtf.gene.probemap", sep="\t", index_col=0)
    probemap.to_csv(OUT / "gene_symbols.tsv", sep="\t")


if __name__ == "__main__":
    main()
