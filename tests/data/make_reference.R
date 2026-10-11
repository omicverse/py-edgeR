# Regenerate the edgeR reference values used by tests/test_r_parity.py.
# Run from the repository root:  Rscript tests/data/make_reference.R
# Reference: R 4.3.3, edgeR 4.0.16, limma 3.58.1, locfit 1.5-9.
suppressMessages(library(edgeR))
set.seed(20261010)
d <- "tests/data"
ng <- 600; np <- 6
lib <- factor(rep(sprintf("lib%d", 1:np), each = 2))
cond <- factor(rep(c("NT", "KO"), np), levels = c("NT", "KO"))
base <- rgamma(ng, 0.6, 0.02)                         # wide abundance range incl. near-zero rows
libfx <- exp(rnorm(np, 0, 0.4))[as.integer(lib)]
fc <- ifelse(runif(ng) < 0.1, sample(c(0.3, 3), ng, TRUE), 1)
mu <- outer(base, libfx) * ifelse(matrix(cond == "KO", ng, 2 * np, byrow = TRUE), fc, 1)
size <- 1 / 0.08
X <- matrix(rnbinom(length(mu), mu = mu, size = size), ng)
X[sample(ng, 30), c(1, 2)] <- 0                        # pairs with all-zero libraries -> zero fits
colnames(X) <- paste(lib, cond, sep = "_"); rownames(X) <- sprintf("g%03d", 1:ng)
write.table(X, file.path(d, "counts.tsv"), sep = "\t", quote = FALSE)
w <- function(x, f) write.table(x, file.path(d, f), sep = "\t", quote = FALSE, row.names = FALSE)

y <- DGEList(X, group = cond); keep <- filterByExpr(y, group = cond); y <- y[keep, , keep.lib.sizes = FALSE]
y <- calcNormFactors(y)
w(data.frame(keep = which(keep) - 1L), "keep.tsv")

# 1) design path: ~ lib + cond
D <- model.matrix(~ lib + cond)
yd <- estimateDisp(y, D)
for (legacy in c(TRUE, FALSE)) {
  f <- glmQLFit(yd, D, legacy = legacy); r <- glmQLFTest(f, coef = ncol(D))
  w(data.frame(logFC = r$table$logFC, logCPM = r$table$logCPM, F = r$table$F, PValue = r$table$PValue,
               s2post = f$var.post, dfprior = rep_len(f$df.prior, nrow(yd))),
    sprintf("design_ql_legacy%d.tsv", legacy))
}
w(data.frame(common = yd$common.dispersion, prior_df = yd$prior.df), "design_disp_scalar.tsv")
w(data.frame(trended = yd$trended.dispersion, tagwise = yd$tagwise.dispersion, ave = yd$AveLogCPM), "design_disp.tsv")

# 2) classic path (no design) + QL + exact test
yc <- estimateDisp(y)
f <- glmQLFit(yc); r <- glmQLFTest(f); e <- exactTest(yc)
w(data.frame(common = yc$common.dispersion, prior_df = yc$prior.df), "classic_disp_scalar.tsv")
w(data.frame(trended = yc$trended.dispersion, tagwise = yc$tagwise.dispersion,
             F = r$table$F, PValue = r$table$PValue, exact_logFC = e$table$logFC, exact_PValue = e$table$PValue),
  "classic.tsv")

# 3) glmFit kernel and Cox-Reid common dispersion
for (k in 1:3) {
  disp <- c(0.001, 0.1, 5)[k]
  g <- glmFit(y, D, dispersion = disp, prior.count = 0)
  w(data.frame(deviance = g$deviance, g$coefficients), sprintf("glmfit_disp%d.tsv", k))
}
w(data.frame(common = estimateGLMCommonDisp(y, D)$common.dispersion), "glm_common.tsv")
