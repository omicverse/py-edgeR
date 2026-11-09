# Reference run: Bioconductor edgeR on every benchmark task.
# Usage: Rscript 02_run_edger.R <tasks_dir> <out_dir> <n_timing_reps> [task_glob]
# Pipelines (edgeR 4.10 defaults):
#   filterByExpr -> normLibSizes (TMM) -> estimateDisp(design)
#   -> glmQLFit -> glmQLFTest   (quasi-likelihood F-test, the recommended pipeline)
#   -> glmFit   -> glmLRT       (likelihood-ratio test)
#   -> exactTest                (classic exact test)
suppressMessages({library(edgeR); library(limma); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
tasks_dir <- args[1]; out_dir <- args[2]; reps <- as.integer(args[3])
pattern <- if (length(args) >= 4) args[4] else "*"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

run_pipeline <- function(X, group) {
  tm <- list()
  t0 <- proc.time()[["elapsed"]]
  y <- DGEList(X, group = group)
  keep <- filterByExpr(y)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- normLibSizes(y)
  design <- model.matrix(~group)
  tm$filter_norm <- proc.time()[["elapsed"]] - t0; t0 <- proc.time()[["elapsed"]]
  y <- estimateDisp(y, design)
  tm$estimateDisp <- proc.time()[["elapsed"]] - t0; t0 <- proc.time()[["elapsed"]]
  qfit <- glmQLFit(y, design)
  ql <- glmQLFTest(qfit, coef = 2)
  tm$QL <- proc.time()[["elapsed"]] - t0; t0 <- proc.time()[["elapsed"]]
  gfit <- glmFit(y, design)
  lrt <- glmLRT(gfit, coef = 2)
  tm$LRT <- proc.time()[["elapsed"]] - t0; t0 <- proc.time()[["elapsed"]]
  et <- exactTest(y)
  tm$exact <- proc.time()[["elapsed"]] - t0
  list(y = y, keep = keep, qfit = qfit, ql = ql, lrt = lrt, et = et, time = tm)
}

tasks <- Sys.glob(file.path(tasks_dir, pattern))
tasks <- tasks[file.exists(file.path(tasks, "counts.tsv.gz"))]
for (td in tasks) {
  name <- basename(td)
  od <- file.path(out_dir, name)
  if (file.exists(file.path(od, "timing.json"))) next
  dir.create(od, showWarnings = FALSE)
  X <- as.matrix(read.delim(gzfile(file.path(td, "counts.tsv.gz")), row.names = 1, check.names = FALSE))
  s <- read.delim(file.path(td, "samples.tsv"), row.names = 1)
  group <- factor(s[colnames(X), "group"], levels = c("Normal", "Tumor"))
  times <- list()
  for (r in seq_len(reps)) {
    res <- run_pipeline(X, group)
    times[[r]] <- res$time
  }
  y <- res$y
  out <- data.frame(
    gene = rownames(y),
    AveLogCPM = y$AveLogCPM,
    trended = y$trended.dispersion,
    tagwise = y$tagwise.dispersion,
    ql_logFC = res$ql$table$logFC,
    ql_F = res$ql$table$F,
    ql_PValue = res$ql$table$PValue,
    ql_FDR = p.adjust(res$ql$table$PValue, "BH"),
    ql_s2post = res$qfit$s2.post,
    lrt_logFC = res$lrt$table$logFC,
    lrt_LR = res$lrt$table$LR,
    lrt_PValue = res$lrt$table$PValue,
    lrt_FDR = p.adjust(res$lrt$table$PValue, "BH"),
    et_logFC = res$et$table$logFC,
    et_PValue = res$et$table$PValue,
    et_FDR = p.adjust(res$et$table$PValue, "BH")
  )
  write.table(out, gzfile(file.path(od, "results.tsv.gz")), sep = "\t", quote = FALSE, row.names = FALSE)
  scal <- list(
    n_keep = sum(res$keep), norm_factors = unname(y$samples$norm.factors),
    common_dispersion = y$common.dispersion, prior_df = y$prior.df,
    ql_df_prior = res$qfit$df.prior, ql_s2_prior = res$qfit$s2.prior,
    ql_average_ql_dispersion = res$qfit$average.ql.dispersion
  )
  write_json(scal, file.path(od, "scalars.json"), digits = NA, auto_unbox = TRUE)
  write_json(list(engine = "edgeR", version = as.character(packageVersion("edgeR")),
                  R = R.version.string, reps = times), file.path(od, "timing.json"), digits = NA, auto_unbox = TRUE)
  cat(name, sprintf("%.2fs", sum(unlist(times[[reps]]))), "\n")
}
