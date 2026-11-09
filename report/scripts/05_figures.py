"""Figures and tables of the pyedger technical report.

Usage: python 05_figures.py <runs_dir> <fgsea.tsv> <out_dir>
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RUNS, FGSEA, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"

# palette (validated reference instance): edgeR = orange, pyedger engines = one blue ramp
C_R = "#eb6834"
C_PY = {"pyedger_numba8": "#104281", "pyedger_numba1": "#6da7ec"}
LBL = {"edgeR": "edgeR 4.10.5 (R, 1 thread)", "pyedger_numba8": "pyedger 0.2.3, 8 threads",
       "pyedger_numba1": "pyedger 0.2.3, 1 thread", "pyedger_numpy": "pyedger 0.2.3, without numba"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f2423"])
QCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})

FDR, LFC = 0.05, 1.0
PYREF = "pyedger_numba1"   # the pyedger run that covers every task
TESTS = {"ql": "QL F-test", "lrt": "LR test", "et": "exact test"}


def load(engine: str, task: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / engine / task / "results.tsv.gz", sep="\t")


def calls(df: pd.DataFrame, test: str) -> np.ndarray:
    sig = (df[f"{test}_FDR"] <= FDR) & (df[f"{test}_logFC"].abs() >= LFC)
    return np.where(sig & (df[f"{test}_logFC"] > 0), 1, np.where(sig, -1, 0))


def rel_err(a, b, floor=1e-250):
    # P-values below 1e-250 count as zero: there R's dnbinom underflows to 0 while SciPy may return a subnormal
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(a), floor)


def label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


full = sorted(p.name for p in (RUNS / "edgeR").glob("full_*"))
cohorts = [t.split("_", 1)[1] for t in full]
subs = sorted(p.name for p in (RUNS / "edgeR").glob("sub*"))

# ---------- per-task comparison table ----------
rows, errs = [], {k: [] for k in ["tagwise", "ql_logFC", "ql_PValue", "lrt_PValue", "et_PValue", "AveLogCPM"]}
for task in full + subs:
    r, p = load("edgeR", task), load(PYREF, task)
    assert (r.gene.to_numpy() == p.gene.to_numpy()).all(), task
    meta = json.load(open(RUNS.parent / "edger_tasks" / task / "meta.json"))
    sr, sp = json.load(open(RUNS / "edgeR" / task / "scalars.json")), json.load(open(RUNS / PYREF / task / "scalars.json"))
    row = dict(task=task, kind=meta["kind"], n=meta.get("n"), samples=meta["n_samples"], genes_kept=len(r),
               nf_maxrel=rel_err(sr["norm_factors"], sp["norm_factors"]).max(),
               common_rel=rel_err([sr["common_dispersion"]], [sp["common_dispersion"]])[0],
               priordf_rel=rel_err([sr["prior_df"]], [sp["prior_df"]])[0],
               qldf_rel=rel_err(np.atleast_1d(sr["ql_df_prior"]), np.atleast_1d(sp["ql_df_prior"])).max())
    for t in TESTS:
        cr, cp = calls(r, t), calls(p, t)
        row[f"{t}_deg_R"], row[f"{t}_deg_py"] = int((cr != 0).sum()), int((cp != 0).sum())
        row[f"{t}_discordant"] = int((cr != cp).sum())
        inter, union = ((cr != 0) & (cr == cp)).sum(), ((cr != 0) | (cp != 0)).sum()
        row[f"{t}_jaccard"] = inter / union if union else 1.0
        row[f"{t}_p_maxrel"] = np.nanmax(rel_err(r[f"{t}_PValue"], p[f"{t}_PValue"]))
        row[f"{t}_lfc_maxabs"] = np.nanmax(np.abs(r[f"{t}_logFC"] - p[f"{t}_logFC"]))
        row[f"{t}_fdr_flip"] = int(((r[f"{t}_FDR"] <= FDR) != (p[f"{t}_FDR"] <= FDR)).sum())
    row["tagwise_maxrel"] = rel_err(r.tagwise, p.tagwise).max()
    rows.append(row)
    if meta["kind"] == "full":
        for k in errs:
            errs[k].append(rel_err(r[k], p[k]))
tab = pd.DataFrame(rows)
# agreement between pyedger runs with 1 and 8 threads
eng = []
for task in full:
    ref = load(PYREF, task)
    for e in ["pyedger_numba8"]:
        if (RUNS / e / task / "results.tsv.gz").exists():
            o = load(e, task)
            eng.append(dict(task=task, engine=e, **{k: np.nanmax(rel_err(ref[k], o[k])) for k in ref.columns[1:]}))
pd.DataFrame(eng).to_csv(OUT / "engine_agreement.tsv", sep="\t", index=False)
tab.to_csv(OUT / "comparison_per_task.tsv", sep="\t", index=False)


# ---------- timing table ----------
def timing(engine, task):
    f = RUNS / engine / task / "timing.json"
    if not f.exists():
        return np.nan
    reps = json.load(open(f))["reps"]
    return float(np.median([sum(r.values()) for r in reps]))


engines = ["edgeR", "pyedger_numba1", "pyedger_numba8"]
tim = pd.DataFrame({e: [timing(e, t) for t in full] for e in engines}, index=cohorts)
tim.insert(0, "samples", [int(tab.set_index("task").loc[t, "samples"]) for t in full])
tim.insert(1, "genes", [int(tab.set_index("task").loc[t, "genes_kept"]) for t in full])
tim.to_csv(OUT / "timing_full.tsv", sep="\t")
stage = {}
for e in engines:
    f = RUNS / e / "full_BRCA" / "timing.json"
    if f.exists():
        stage[e] = pd.DataFrame(json.load(open(f))["reps"]).median()
pd.DataFrame(stage).to_csv(OUT / "timing_stages_BRCA.tsv", sep="\t")

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.9))
gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.0], hspace=0.38)


def panel(ax, s, dx=-0.02):
    ax.text(dx, 1.03, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


# row 1: logos (top-left) + A: DE-call confusion matrices of the QL F-test
row1 = gs[0].subgridspec(1, 2, width_ratios=[0.16, 1.0], wspace=0.04)
logo = row1[0].subgridspec(2, 1, height_ratios=[1.25, 0.55], hspace=0.0)
for k, f in enumerate(["pyedger_logo.png", "omicverse_logo.png"]):
    axl = fig.add_subplot(logo[k])
    axl.imshow(plt.imread(ASSETS / f))
    axl.set_axis_off()
top = row1[1].subgridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.12, hspace=0.38)
cats = [1, 0, -1]
names = ["up", "ns", "dn"]
for i, (task, coh) in enumerate(zip(full, cohorts)):
    ax = fig.add_subplot(top[i // 4, i % 4])
    cr, cp = calls(load("edgeR", task), "ql"), calls(load(PYREF, task), "ql")
    M = np.array([[((cr == a) & (cp == b)).sum() for b in cats] for a in cats], float)
    P = M / np.maximum(M.sum(1, keepdims=True), 1) * 100
    im = ax.imshow(P, cmap=SEQ, vmin=0, vmax=100)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{int(M[a, b])}", ha="center", va="center", fontsize=4.5,
                    color="white" if P[a, b] > 55 else INK)
    ax.set_xticks(range(3), names if i // 4 == 1 else [""] * 3, fontsize=5)
    ax.set_yticks(range(3), names if i % 4 == 0 else [""] * 3, fontsize=5)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{coh} (n={int(tab.set_index('task').loc[task, 'samples'])})", fontsize=6, pad=2)
    if i % 4 == 0:
        ax.set_ylabel("edgeR", fontsize=6, labelpad=1)
    if i // 4 == 1:
        ax.set_xlabel("pyedger", fontsize=6, labelpad=1)
    if i == 0:
        panel(ax, "a", dx=-0.32)
cax = fig.add_subplot(top[:, 4])
cb = fig.colorbar(im, cax=cax)
cb.set_label("% of edgeR row", fontsize=5.5)
cb.ax.tick_params(labelsize=5)
cb.outline.set_visible(False)

# row 2: B (pathways) + C (per-gene numerical agreement)
mid = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.28)
fg = pd.read_csv(FGSEA, sep="\t")
fg["cohort"] = fg.task.str.replace("full_", "", regex=False)
nesw = fg.pivot_table(index=["task", "pathway"], columns="engine", values="NES")
axb = fig.add_subplot(mid[0])
axb.axhline(0, color=INK2, lw=0.6)
axb.scatter(nesw["edgeR"], nesw["edgeR_seed2"] - nesw["edgeR"], s=7, color="#a3a29d", edgecolor="white", linewidth=0.3,
            label="edgeR, other fgsea seed")
axb.scatter(nesw["edgeR"], nesw["pyedger"] - nesw["edgeR"], s=7, color=C_PY["pyedger_numba1"], edgecolor="white",
            linewidth=0.3, label="pyedger, same seed")
axb.set_xlabel("NES from edgeR statistics (Hallmark, 8 cohorts)")
axb.set_ylabel("NES difference from edgeR")
axb.grid(color=GRID, lw=0.5)
axb.set_axisbelow(True)
dmax_py = (nesw["pyedger"] - nesw["edgeR"]).abs().max()
dmax_seed = (nesw["edgeR_seed2"] - nesw["edgeR"]).abs().max()
axb.legend(fontsize=5, loc="upper left", handletextpad=0.2, borderaxespad=0.3)
axb.text(0.0, -0.062, f"max |\u0394|\npyedger {dmax_py:.0e}\nother seed {dmax_seed:.2f}",
         ha="center", va="center", fontsize=5, color=INK2)
panel(axb, "b", dx=-0.16)

axc = fig.add_subplot(mid[1])
qnames = {"AveLogCPM": "AveLogCPM", "tagwise": "tagwise dispersion", "ql_logFC": "logFC (QL)",
          "ql_PValue": "P (QL F-test)", "lrt_PValue": "P (LR test)", "et_PValue": "P (exact test)"}
for k, col in zip(qnames, QCOL):
    e = np.log10(np.maximum(np.concatenate(errs[k]), 1e-17))
    xs = np.sort(e)
    axc.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, lw=1.2, label=f"{qnames[k]} (max {10 ** xs[-1]:.0e})")
axc.axvline(-6, color=INK2, lw=0.6, ls=(0, (2, 2)))
axc.text(-5.85, 0.62, "unit-test\ntolerance", fontsize=5, color=INK2, va="center")
axc.set_xlabel("log$_{10}$ |pyedger − edgeR| / |edgeR|  (per gene, 8 cohorts)")
axc.set_ylabel("cumulative fraction of genes")
axc.set_xlim(-17, -2)
axc.grid(color=GRID, lw=0.5)
axc.legend(fontsize=4.8, loc="lower right", handlelength=1.4, borderaxespad=0.3)
panel(axc, "c", dx=-0.1)

# row 3: D (runtime) + E (small samples)
bot = gs[2].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.24)
axd = fig.add_subplot(bot[0])
x = np.arange(len(cohorts))
w = 0.27
for k, e in enumerate(engines):
    axd.bar(x + (k - 1) * w, tim[e].to_numpy(), w * 0.9, color=C_R if e == "edgeR" else C_PY[e], label=LBL[e])
axd.set_xticks(x, [f"{c}\n({n})" for c, n in zip(cohorts, tim.samples)], fontsize=5)
axd.set_ylabel("wall time (s)")
axd.grid(axis="y", color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=5, loc="upper right", borderaxespad=0.2)
panel(axd, "d", dx=-0.08)

axe = fig.add_subplot(bot[1])
ss = tab[tab.kind == "sub"].copy()
for k, (t, col) in enumerate(zip(TESTS, [QCOL[0], QCOL[2], QCOL[4]])):
    for i, n in enumerate(sorted(ss.n.unique())):
        v = ss.loc[ss.n == n, f"{t}_p_maxrel"].to_numpy()
        jit = np.random.default_rng(i).uniform(-0.07, 0.07, len(v))
        axe.scatter(np.full(len(v), i + (k - 1) * 0.25) + jit, np.log10(np.maximum(v, 1e-17)), s=6, color=col,
                    edgecolor="white", linewidth=0.3, label=TESTS[t] if i == 0 else None)
axe.axhline(-6, color=INK2, lw=0.6, ls=(0, (2, 2)))
axe.set_xticks(range(3), [f"{int(n)} vs {int(n)}" for n in sorted(ss.n.unique())])
axe.set_ylabel("log$_{10}$ max rel. difference of P")
axe.set_xlabel("TCGA-LUAD, normal vs tumour (20 draws each)")
axe.set_ylim(-9, -5.2)
axe.grid(axis="y", color=GRID, lw=0.5)
axe.legend(fontsize=5, loc="upper right", ncol=3, columnspacing=0.6, handletextpad=0.1, borderaxespad=0.2)
panel(axe, "e", dx=-0.12)

fig.savefig(OUT / "fig1.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1.png", bbox_inches="tight", dpi=200)

# ======================= Figure S1: -log10 P scatter per cohort =======================
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8))
for ax, task, coh in zip(axs.flat, full, cohorts):
    r, p = load("edgeR", task), load(PYREF, task)
    a = -np.log10(np.maximum(r.ql_PValue, 1e-320))
    b = -np.log10(np.maximum(p.ql_PValue, 1e-320))
    ax.scatter(a, b, s=1.5, color=C_PY["pyedger_numba1"], linewidth=0, rasterized=True)
    lim = max(a.max(), b.max()) * 1.03
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.5)
    ax.set_title(f"TCGA-{coh}", fontsize=6, pad=2)
    ax.text(0.04, 0.9, f"r = {np.corrcoef(a, b)[0, 1]:.12f}", transform=ax.transAxes, fontsize=5)
for ax in axs[1]:
    ax.set_xlabel("edgeR −log$_{10}$P (QL)")
for ax in axs[:, 0]:
    ax.set_ylabel("pyedger −log$_{10}$P (QL)")
fig.tight_layout()
fig.savefig(OUT / "figS1_pvalue_scatter.pdf", bbox_inches="tight")
fig.savefig(OUT / "figS1_pvalue_scatter.png", bbox_inches="tight", dpi=200)
print(tab[["task", "samples", "genes_kept", "ql_deg_R", "ql_deg_py", "ql_discordant", "ql_fdr_flip", "ql_p_maxrel", "tagwise_maxrel"]].to_string())
print(tim.to_string())


# ======================= numbers quoted in the text (LaTeX macros) =======================
def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


F, S = tab[tab.kind == "full"], tab[tab.kind == "sub"]
fgw = fg.pivot_table(index=["task", "pathway"], columns="engine", values="padj")
nes = fg.pivot_table(index=["task", "pathway"], columns="engine", values="NES")
sig = fgw <= 0.05
agree = pd.read_csv(OUT / "engine_agreement.tsv", sep="\t")
num_cols = [c for c in agree.columns if c.endswith("PValue") or c in ("tagwise", "trended", "AveLogCPM")]
nb = tim.loc["BRCA"]
macros = {
    "nCohorts": len(full), "nSamplesTotal": f"{int(F.samples.sum()):,}",
    "nSamplesMin": int(F.samples.min()), "nSamplesMax": f"{int(F.samples.max()):,}",
    "nGenesMin": f"{int(F.genes_kept.min()):,}", "nGenesMax": f"{int(F.genes_kept.max()):,}",
    "qlDegTotal": f"{int(F.ql_deg_R.sum()):,}", "qlDiscordTotal": int(F.ql_discordant.sum()),
    "lrtDiscordTotal": int(F.lrt_discordant.sum()), "etDiscordTotal": int(F.et_discordant.sum()),
    "fdrFlipTotal": int(F[["ql_fdr_flip", "lrt_fdr_flip", "et_fdr_flip"]].to_numpy().sum()),
    "qlPmaxrel": sci(F.ql_p_maxrel.max()), "lrtPmaxrel": sci(F.lrt_p_maxrel.max()), "etPmaxrel": sci(F.et_p_maxrel.max()),
    "tagwiseMaxrel": sci(F.tagwise_maxrel.max()), "nfMaxrel": sci(tab.nf_maxrel.max()),
    "commonMaxrel": sci(tab.common_rel.max()), "priordfMaxrel": sci(tab.priordf_rel.max()),
    "qlLfcMaxabs": sci(F.ql_lfc_maxabs.max()),
    "qlPmedrel": sci(float(np.median(np.concatenate(errs["ql_PValue"])))),
    "nSubTasks": len(S), "subPmaxrel": sci(S[["ql_p_maxrel", "lrt_p_maxrel", "et_p_maxrel"]].to_numpy().max()),
    "subDiscordTotal": int(S[["ql_discordant", "lrt_discordant", "et_discordant"]].to_numpy().sum()),
    "subJaccardMin": f"{S[['ql_jaccard', 'lrt_jaccard', 'et_jaccard']].to_numpy().min():.4f}",
    "fgseaPairs": int(sig["edgeR"].size), "fgseaSigR": int(sig["edgeR"].sum()),
    "fgseaDiscordPy": int((sig["edgeR"] != sig["pyedger"]).sum()),
    "fgseaDiscordSeed": int((sig["edgeR"] != sig["edgeR_seed2"]).sum()),
    "fgseaNesMaxabsPy": sci((nes["edgeR"] - nes["pyedger"]).abs().max()),
    "fgseaNesMaxabsSeed": f"{(nes['edgeR'] - nes['edgeR_seed2']).abs().max():.3f}",
    "engineMaxrel": sci(agree[num_cols].to_numpy().max()) if len(agree) else "n/a",
    "tBRCAR": f"{nb['edgeR']:.0f}", "tBRCAnumbaOne": f"{nb['pyedger_numba1']:.0f}",
    "tBRCAnumbaEight": f"{nb['pyedger_numba8']:.0f}",
    "spEightMin": f"{(tim.edgeR / tim.pyedger_numba8).min():.1f}", "spEightMax": f"{(tim.edgeR / tim.pyedger_numba8).max():.1f}",
    "rtNumbaOneMin": f"{(tim.pyedger_numba1 / tim.edgeR).min():.2f}", "rtNumbaOneMax": f"{(tim.pyedger_numba1 / tim.edgeR).max():.2f}",
    "rtNumbaEightMin": f"{(tim.pyedger_numba8 / tim.edgeR).min():.2f}", "rtNumbaEightMax": f"{(tim.pyedger_numba8 / tim.edgeR).max():.2f}",
}
# the same benchmark with pyedger 0.2.2 (before the fixes), if present
OLD = RUNS.parent / "edger_runs_v022"
if (OLD / "pyedger_numba1").exists():
    flips, bad = 0, 0
    for task in full + subs:
        r, o = load("edgeR", task), pd.read_csv(OLD / "pyedger_numba1" / task / "results.tsv.gz", sep="\t")
        flips += int(((r.et_FDR <= FDR) != (o.et_FDR <= FDR)).sum())
        bad += int((np.abs(np.log10(r.et_PValue.clip(1e-320)) - np.log10(o.et_PValue.clip(1e-320))) > 0.01).sum())
    told = pd.DataFrame({e: [json.load(open(OLD / e / t / "timing.json"))["reps"][0] for t in full] for e in
                         ["pyedger_numba1", "pyedger_numba8"]}, index=cohorts).map(lambda d: sum(d.values()))
    macros.update({
        "oldEtFlip": flips, "oldEtBad": bad,
        "oldRtNumbaOneMin": f"{(told.pyedger_numba1 / tim.edgeR).min():.2f}", "oldRtNumbaOneMax": f"{(told.pyedger_numba1 / tim.edgeR).max():.2f}",
        "oldRtNumbaEightMin": f"{(told.pyedger_numba8 / tim.edgeR).min():.2f}", "oldRtNumbaEightMax": f"{(told.pyedger_numba8 / tim.edgeR).max():.2f}",
        "oldBRCAnumbaEight": f"{told.loc['BRCA', 'pyedger_numba8']:.0f}",
    })
    told.to_csv(OUT / "timing_full_v022.tsv", sep="\t")
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
