# Regenerate the edgeR reference values used by tests/test_r_parity.py.
# Run from the repository root:  Rscript tests/data/make_reference.R
# Reference: R 4.5.3, edgeR 4.10.5, limma 3.68.5, locfit 1.5-9.12.
#
# counts.tsv was simulated once (R 4.3.3) with:
#   set.seed(20261010); ng <- 600; np <- 6
#   lib <- factor(rep(sprintf("lib%d", 1:np), each = 2))
#   cond <- factor(rep(c("NT", "KO"), np), levels = c("NT", "KO"))
#   base <- rgamma(ng, 0.6, 0.02); libfx <- exp(rnorm(np, 0, 0.4))[as.integer(lib)]
#   fc <- ifelse(runif(ng) < 0.1, sample(c(0.3, 3), ng, TRUE), 1)
#   mu <- outer(base, libfx) * ifelse(matrix(cond == "KO", ng, 2 * np, byrow = TRUE), fc, 1)
#   X <- matrix(rnbinom(length(mu), mu = mu, size = 1 / 0.08), ng)
#   X[sample(ng, 30), c(1, 2)] <- 0        # pairs with all-zero libraries -> zero fits
# It is read back here so the reference does not depend on the RNG of the R version.
suppressMessages({library(edgeR); library(limma)})
d <- "tests/data"
w <- function(x, f) write.table(x, file.path(d, f), sep = "\t", quote = FALSE, row.names = FALSE)
X <- as.matrix(read.delim(file.path(d, "counts.tsv"), row.names = 1))
lib <- factor(sub("_(KO|NT)$", "", colnames(X)))
cond <- factor(sub("^.*_", "", colnames(X)), levels = c("NT", "KO"))

y <- DGEList(X, group = cond); keep <- filterByExpr(y, group = cond); y <- y[keep, , keep.lib.sizes = FALSE]
y <- normLibSizes(y)
w(data.frame(keep = which(keep) - 1L), "keep.tsv")
w(data.frame(norm = y$samples$norm.factors), "norm_factors.tsv")

ql <- function(f, r, file) {
  w(data.frame(logFC = r$table$logFC, logCPM = r$table$logCPM, F = r$table$F, PValue = r$table$PValue,
               s2post = f$s2.post, s2prior = rep_len(f$s2.prior, nrow(f)), dfprior = rep_len(f$df.prior, nrow(f)),
               dfresid = if (is.null(f$df.residual.adj)) f$df.residual.zeros else f$df.residual.adj), file)
  if (!is.null(f$top.proportion))
    w(data.frame(top_proportion = f$top.proportion, dispersion = f$dispersion[1],
                 ave_ql = f$average.ql.dispersion), sub(".tsv", "_scalar.tsv", file, fixed = TRUE))
}

# 1) design path: ~ lib + cond
D <- model.matrix(~ lib + cond)
yd <- estimateDisp(y, D)
w(data.frame(common = yd$common.dispersion, prior_df = yd$prior.df, span = yd$span), "design_disp_scalar.tsv")
w(data.frame(trended = yd$trended.dispersion, tagwise = yd$tagwise.dispersion, ave = yd$AveLogCPM), "design_disp.tsv")
yl <- estimateDisp(y, D, legacy.span = TRUE)
w(data.frame(common = yl$common.dispersion, prior_df = yl$prior.df, span = yl$span), "design_disp_legacyspan_scalar.tsv")
w(data.frame(trended = yl$trended.dispersion, tagwise = yl$tagwise.dispersion), "design_disp_legacyspan.tsv")
for (legacy in c(TRUE, FALSE)) {
  f <- glmQLFit(yd, D, legacy = legacy)
  ql(f, glmQLFTest(f, coef = ncol(D)), sprintf("design_ql_legacy%d.tsv", legacy))
}
# new QL straight from counts (no dispersions in the DGEList)
f <- glmQLFit(y, D)
ql(f, glmQLFTest(f, coef = ncol(D)), "design_ql_nodisp.tsv")
f <- glmQLFit(yd, D, top.proportion = 0.2, abundance.trend = FALSE)
ql(f, glmQLFTest(f, coef = ncol(D)), "design_ql_top02_notrend.tsv")

# 2) classic path (no design) + QL + exact test
yc <- estimateDisp(y)
w(data.frame(common = yc$common.dispersion, prior_df = yc$prior.df, span = yc$span), "classic_disp_scalar.tsv")
f1 <- glmQLFit(yc, legacy = TRUE); r1 <- glmQLFTest(f1)
f0 <- glmQLFit(yc); r0 <- glmQLFTest(f0)
e <- exactTest(yc)
w(data.frame(trended = yc$trended.dispersion, tagwise = yc$tagwise.dispersion,
             F_legacy = r1$table$F, PValue_legacy = r1$table$PValue, F = r0$table$F, PValue = r0$table$PValue,
             exact_logFC = e$table$logFC, exact_PValue = e$table$PValue), "classic.tsv")

# 3) glmFit kernel and Cox-Reid common dispersion
for (k in 1:3) {
  disp <- c(0.001, 0.1, 5)[k]
  g <- glmFit(y, D, dispersion = disp, prior.count = 0)
  w(data.frame(deviance = g$deviance, g$coefficients), sprintf("glmfit_disp%d.tsv", k))
}
w(data.frame(common = estimateGLMCommonDisp(y, D)$common.dispersion), "glm_common.tsv")

# 4) Milo configuration (omicverse Milo.da_nhoods): trend.method = "none", new QL
Y <- as.matrix(read.delim(file.path(d, "milo_counts.tsv"), row.names = 1))
Dm <- cbind(1, c(0, 0, 0, 0, 1, 1, 1, 1))
dm <- DGEList(Y); dm$samples$norm.factors <- 1
dm <- estimateDisp(dm, Dm, trend.method = "none", robust = FALSE)
f <- glmQLFit(dm, Dm, robust = FALSE, legacy = FALSE)
ql(f, glmQLFTest(f, coef = 2), "milo_ql.tsv")
w(data.frame(common = dm$common.dispersion, prior_df = dm$prior.df), "milo_disp_scalar.tsv")
w(data.frame(tagwise = dm$tagwise.dispersion), "milo_disp.tsv")

# 5) Single-value regressions: TMM on tied log-ratios, infinite prior df
p <- read.delim(file.path(d, "tmm_ties.tsv.gz"))
w(data.frame(factor = sprintf("%.17g", edgeR:::.calcFactorTMM(rep(p$obs, p$n), rep(p$ref, p$n)))), "tmm_ties_ref.tsv")
nx <- read.delim(file.path(d, "no_extra_var.tsv"))
yn <- estimateDisp(normLibSizes(DGEList(as.matrix(nx[, 1:6]), group = factor(rep(c("treatment", "control"), each = 3)))))
w(data.frame(prior_df = yn$prior.df, tagwise = yn$tagwise.dispersion,
             PValue = glmLRT(glmFit(yn), coef = 2)$table$PValue), "no_extra_var_ref.tsv")

# 6) limma 3.68 hyperparameter estimation (robust / unequal df / covariate)
set.seed(3); n <- 2000; cov <- rnorm(n)
x <- exp(0.3 * cov) * rchisq(n, 4) / 4 / (rchisq(n, 8) / 8); x[1:20] <- x[1:20] * 30
df1u <- sample(c(3, 4), n, TRUE)
w(data.frame(x = sprintf("%.17g", x), cov = sprintf("%.17g", cov), df1u = df1u), "limma_in.tsv")
o <- list(fitFDistRobustly(x, 4), fitFDistRobustly(x, 4, covariate = cov), fitFDistRobustly(x, df1u),
          squeezeVar(x, df1u, robust = TRUE), squeezeVar(x, df1u, covariate = cov, robust = TRUE), squeezeVar(x, df1u, covariate = cov))
w(data.frame(s1 = rep_len(o[[1]]$scale, n), d1 = o[[1]]$df2.shrunk, s2 = o[[2]]$scale, d2 = o[[2]]$df2.shrunk,
             s3 = rep_len(o[[3]]$scale, n), d3 = o[[3]]$df2.shrunk,
             p4 = o[[4]]$var.post, d4 = rep_len(o[[4]]$df.prior, n), p5 = o[[5]]$var.post, d5 = rep_len(o[[5]]$df.prior, n),
             p6 = o[[6]]$var.post, d6 = rep_len(o[[6]]$df.prior, n)), "limma_out.tsv")
