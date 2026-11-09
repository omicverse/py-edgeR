# pyedger technical report: benchmark

The report is published at <https://omicverse.org/articles/pyedger.html>. It compares
pyedger with Bioconductor edgeR 4.10.5 on eight TCGA tumour–normal cohorts and 60
small-sample draws. This directory holds the benchmark that produces its numbers and figures.

## Reproduce

Requirements: a Python env with `pyedger`, pandas, matplotlib; an R env with
edgeR 4.10.5, limma 3.68.5, fgsea, jsonlite, data.table (e.g. bioconda
`r-base=4.5 bioconductor-edger=4.10.5 bioconductor-limma=3.68.5 bioconductor-fgsea r-jsonlite`);
and PyMuPDF to export the figures for the article.

```bash
DATA=/path/to/data            # ~1 GB of downloads + ~1.5 GB of tasks
mkdir -p $DATA/tcga $DATA/genesets
for p in BRCA COAD KIRC LIHC LUAD LUSC PRAD THCA; do
  curl -fLo $DATA/tcga/TCGA-$p.star_counts.tsv.gz https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-$p.star_counts.tsv.gz
done
curl -fLo $DATA/tcga/gencode.v36.annotation.gtf.gene.probemap https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap
curl -fLo $DATA/genesets/MSigDB_Hallmark_2020.gmt "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"

python scripts/01_prepare_tcga.py $DATA                                   # -> $DATA/edger_tasks
TASKS=$DATA/edger_tasks OUT=$DATA/edger_runs scripts/run_all.sh          # edgeR, pyedger 1 and 8 threads
Rscript scripts/04_fgsea.R $DATA/edger_runs $DATA/edger_tasks/gene_symbols.tsv \
        $DATA/genesets/MSigDB_Hallmark_2020.gmt $DATA/edger_runs/fgsea_hallmark.tsv
python scripts/05_figures.py $DATA/edger_runs $DATA/edger_runs/fgsea_hallmark.tsv figures
```

`05_figures.py` writes `figures/numbers.tex`, the numbers quoted in the article.

| File | Content |
|---|---|
| `figures/fig1.pdf` | main figure (DE calls, pathways, numerical agreement, speed, small samples) |
| `figures/figS1_pvalue_scatter.pdf` | −log10 P, edgeR vs pyedger, per cohort |
| `figures/comparison_per_task.tsv` | per-task parity metrics (68 tasks) |
| `figures/timing_full.tsv`, `timing_stages_BRCA.tsv` | wall times |
| `figures/engine_agreement.tsv` | pyedger 1 vs 8 threads (bit-identical) |
