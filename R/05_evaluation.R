# =============================================================================
# 05_evaluation.R — Full 9-Metric Evaluation Suite + Statistical Tests
# BioCluster-R | ICCA 2026
#
# Inputs  : data/processed/corpus_balanced.rds
#           data/umap/umap50_tfidf.rds
#           data/umap/umap50_pubmedbert.rds
#           data/clustering/all_cluster_assignments.rds
#           data/clustering/sil_subsample_idx.rds
#           data/clustering/hdbscan_meta.rds
#
# Outputs : results/table3.rds/.csv             ← Table 3 (paper)
#           results/per_class_f1.rds/.csv        ← per-class F1 detail
#           results/significance_tests.rds
#           results/significance_summary.rds/.csv
#
# NAMESPACE NOTE: clusterSim::select() conflicts with dplyr::select().
# Use dplyr::select() explicitly throughout this script.
#
# COLUMN NOTE: per_class_f1 intentionally stores disease_name via DISEASE_MAP
# inside the loop. The left_join on corpus labels is NOT done here — that
# would create disease_name.x/disease_name.y conflicts. 06_keywords.R uses
# the class integer to look up F1, avoiding string matching entirely.
#
# Next    : 06_keywords.R
# =============================================================================

suppressPackageStartupMessages({
  library(clusterSim)   # index.DB() — load BEFORE dplyr
  library(cluster)
  library(aricode)
  library(clue)
})
source("R/utils.R")
suppressPackageStartupMessages({ library(tidyverse) })

set.seed(42)
t0 <- proc.time()
cat("── 05_evaluation.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# =============================================================================
# Section 0: Data Import and Configuration
# Loads the evaluated corpus, generated UMAP projections, cluster assignments, 
# and stratified silhouette indices required for comprehensive metric calculation.
# =============================================================================
cat("0. Loading inputs...\n")
corpus      <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))
true_labels <- corpus$label
corpus_ids  <- corpus$doc_id
stopifnot(nrow(corpus) == 7470)

umap50 <- list(
  tfidf      = as.matrix(readRDS(file.path(PATHS$umap, "umap50_tfidf.rds"))),
  pubmedbert = as.matrix(readRDS(file.path(PATHS$umap, "umap50_pubmedbert.rds")))
)
all_cluster_assignments <- readRDS(file.path(PATHS$clustering, "all_cluster_assignments.rds"))
sil_idx      <- readRDS(file.path(PATHS$clustering, "sil_subsample_idx.rds"))
hdbscan_meta <- readRDS(file.path(PATHS$clustering, "hdbscan_meta.rds"))

stopifnot(length(all_cluster_assignments) == 6L, length(sil_idx) == 2000L)
cat(sprintf("   corpus: %d | experiments: %d | sil_subsample: %d ✓\n\n",
            nrow(corpus), length(all_cluster_assignments), length(sil_idx)))

# =============================================================================
# Section 1: Algorithmic Evaluation Utilities
# Implements bespoke metric computations including optimal Hungarian assignment 
# for Macro-F1 and robust V-Measure calculations as formalized in Methodology §3.7.
# =============================================================================

compute_purity <- function(pred, true) {
  ct <- table(cluster = pred, class = true)
  round(sum(apply(ct, 1, max)) / length(pred), 4)
}

# --- Optimal Assignment Macro-F1 Evaluation ----------------------------------
compute_macro_f1 <- function(pred, true) {
  classes  <- 1:5
  clusters <- sort(unique(pred))
  n_cls    <- length(classes)
  n_clu    <- length(clusters)
  
  ct <- matrix(0L, nrow = n_cls, ncol = max(n_cls, n_clu))
  for (i in seq_along(classes))
    for (j in seq_along(clusters))
      ct[i, j] <- sum(true == classes[i] & pred == clusters[j])
  
  assignment <- clue::solve_LSAP(ct, maximum = TRUE)
  
  per_class <- lapply(seq_along(classes), function(i) {
    j   <- assignment[i]
    cls <- classes[i]
    if (j > n_clu)
      return(data.frame(class = cls, tp = 0L, fp = 0L, fn = sum(true == cls),
                        precision = 0, recall = 0, f1 = 0))
    ac   <- clusters[j]
    tp   <- ct[i, j]
    fp   <- sum(pred == ac) - tp
    fn   <- sum(true == cls) - tp
    prec <- if (tp + fp == 0L) 0 else tp / (tp + fp)
    rec  <- if (tp + fn == 0L) 0 else tp / (tp + fn)
    f1   <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
    data.frame(class = cls, tp = tp, fp = fp, fn = fn,
               precision = round(prec, 4),
               recall    = round(rec,  4),
               f1        = round(f1,   4))
  })
  per_class_df <- bind_rows(per_class)
  list(macro_f1 = round(mean(per_class_df$f1), 4), per_class = per_class_df)
}

# --- Information Theoretic Metrics (Homogeneity, Completeness, V-Measure) ----
# Computes positive entropy boundaries using weighted conditionals.
compute_h_vmeasure <- function(pred, true) {
  n  <- length(pred)
  ct <- table(cluster = pred, class = true)
  
  safe_ent <- function(p) { p <- p[p > 0]; -sum(p * log(p)) }
  
  H_C <- safe_ent(colSums(ct) / n)
  H_K <- safe_ent(rowSums(ct) / n)
  
  H_C_given_K <- sum(sapply(seq_len(nrow(ct)), function(i) {
    pk <- sum(ct[i, ]) / n
    if (pk == 0) return(0)
    pk * safe_ent(ct[i, ] / sum(ct[i, ]))
  }))
  H_K_given_C <- sum(sapply(seq_len(ncol(ct)), function(j) {
    pc <- sum(ct[, j]) / n
    if (pc == 0) return(0)
    pc * safe_ent(ct[, j] / sum(ct[, j]))
  }))
  
  hom  <- if (H_C == 0) 1.0 else 1 - H_C_given_K / H_C
  comp <- if (H_K == 0) 1.0 else 1 - H_K_given_C / H_K
  vmea <- if (hom + comp == 0) 0 else 2 * hom * comp / (hom + comp)
  
  list(homogeneity  = round(max(0, min(1, hom)),  4),
       completeness = round(max(0, min(1, comp)), 4),
       v_measure    = round(max(0, min(1, vmea)), 4))
}

# =============================================================================
# Section 2: Comprehensive Experimental Evaluation 
# Iterates through all 6 experimental configurations, computing 9 internal 
# and external cluster validation metrics as outlined in Methodology §3.7.
# =============================================================================
cat("2. Evaluation loop (6 experiments)...\n\n")

results_list  <- list()
per_class_all <- list()

for (i in seq_along(all_cluster_assignments)) {
  exp       <- all_cluster_assignments[[i]]
  exp_name  <- exp$experiment
  repr_name <- exp$repr
  alg_name  <- exp$algorithm
  pred_all  <- exp$labels
  mat       <- umap50[[repr_name]]
  
  cat(sprintf("   [%d/6] %s + %s\n", i, toupper(repr_name), toupper(alg_name)))
  
  is_hdbscan <- alg_name == "hdbscan"
  noise_mask <- pred_all == 0L
  n_noise    <- sum(noise_mask)
  noise_pct  <- round(100 * n_noise / 7470, 1)
  
  if (is_hdbscan) {
    keep_mask <- !noise_mask
    pred_eval <- pred_all[keep_mask]
    true_eval <- true_labels[keep_mask]
    mat_eval  <- mat[keep_mask, ]
    cat(sprintf("     Noise excluded: %d docs (%.1f%%)\n", n_noise, noise_pct))
  } else {
    pred_eval <- pred_all; true_eval <- true_labels; mat_eval <- mat
    n_noise <- 0L; noise_pct <- 0.0
  }
  
  k_used <- length(unique(pred_eval))
  
  # External metrics
  ari      <- round(ARI(pred_eval, true_eval), 4)
  nmi      <- round(NMI(pred_eval, true_eval), 4)
  f1_res   <- compute_macro_f1(pred_eval, true_eval)
  macro_f1 <- f1_res$macro_f1
  hv       <- compute_h_vmeasure(pred_eval, true_eval)
  purity   <- compute_purity(pred_eval, true_eval)
  
  # Caches per-class F1 scores mapped explicitly via integer IDs to preserve 
  # pipeline stability and prevent string matching conflicts downstream.
  per_class_all[[exp_name]] <- f1_res$per_class %>%
    mutate(experiment   = exp_name,
           disease_name = DISEASE_MAP[as.character(class)])
  
  cat(sprintf("     ARI=%.4f | NMI=%.4f | F1=%.4f | Hom=%.4f | V=%.4f | Purity=%.4f\n",
              ari, nmi, macro_f1, hv$homogeneity, hv$v_measure, purity))
  
  # Silhouette on stratified subsample
  sil_eval  <- if (is_hdbscan) intersect(sil_idx, which(!noise_mask)) else sil_idx
  sil_score <- if (length(unique(pred_all[sil_eval])) >= 2L && length(sil_eval) >= 10L)
    round(mean(silhouette(pred_all[sil_eval], dist(mat[sil_eval, ]))[, 3]), 4) else NA_real_
  
  # DBI — clusterSim on full evaluated set
  dbi_score <- tryCatch(
    round(clusterSim::index.DB(mat_eval, pred_eval)$DB, 4),
    error = function(e) { cat(sprintf("     DBI NA: %s\n", e$message)); NA_real_ }
  )
  
  cat(sprintf("     Silhouette=%.4f | DBI=%.4f | k=%d\n\n",
              sil_score, dbi_score, k_used))
  
  results_list[[exp_name]] <- data.frame(
    Experiment     = exp_name,
    Representation = repr_name,
    Algorithm      = alg_name,
    k              = k_used,
    ARI            = ari,
    NMI            = nmi,
    Macro_F1       = macro_f1,
    Homogeneity    = hv$homogeneity,
    V_measure      = hv$v_measure,
    Purity         = purity,
    Silhouette     = sil_score,
    DBI            = dbi_score,
    Noise_pct      = noise_pct,
    stringsAsFactors = FALSE
  )
}

metrics_df   <- bind_rows(results_list)
per_class_df <- bind_rows(per_class_all)   # disease_name already added in loop — no join needed

# =============================================================================
# Section 3: Aggregation of Main Results (Table 3)
# Formats and serializes the primary quantitative findings comparing 
# representation and algorithmic efficacy.
# =============================================================================
cat("3. Assembling Table 3...\n")

REPR_DISPLAY <- c(tfidf = "TF-IDF", pubmedbert = "PubMedBERT")
ALG_DISPLAY  <- c(kmeans = "K-Means", hdbscan = "HDBSCAN",
                  agglomerative = "Agglomerative")

table3 <- metrics_df %>%
  mutate(Representation = REPR_DISPLAY[Representation],
         Algorithm      = ALG_DISPLAY[Algorithm]) %>%
  arrange(match(Representation, c("TF-IDF", "PubMedBERT")),
          match(Algorithm, c("K-Means", "HDBSCAN", "Agglomerative"))) %>%
  dplyr::select(Representation, Algorithm, k,
                ARI, NMI, Macro_F1, Homogeneity, V_measure,
                Purity, Silhouette, DBI, Noise_pct)

cat("\n   ── TABLE 3 ──\n")
print(table3, digits = 4)

best_row      <- metrics_df[which.max(metrics_df$ARI), ]
best_exp_name <- best_row$Experiment
cat(sprintf("\n   Best by ARI: %s (ARI=%.4f, MacroF1=%.4f)\n\n",
            best_exp_name, best_row$ARI, best_row$Macro_F1))

save_artifact(table3,       file.path(PATHS$results, "table3"))
save_artifact(per_class_df, file.path(PATHS$results, "per_class_f1"))

# =============================================================================
# Section 4: Rigorous Statistical Significance Testing
# Performs permutation and non-parametric rank tests to objectively validate 
# the superiority of contextual embeddings over traditional representations.
# =============================================================================
cat("4. Statistical significance tests...\n\n")
sig_results <- list()

# --- Test 1: Permutation Test for ARI Significance (B=1000) ------------------
cat("   Test 1: Permutation test on ARI (B=1000)...\n")
set.seed(42)
perm_rows <- lapply(seq_along(all_cluster_assignments), function(i) {
  exp      <- all_cluster_assignments[[i]]
  pred_all <- exp$labels
  keep     <- if (exp$algorithm == "hdbscan") pred_all != 0L else rep(TRUE, 7470L)
  pred_ev  <- pred_all[keep]; true_ev <- true_labels[keep]
  obs_ari  <- metrics_df$ARI[metrics_df$Experiment == exp$experiment]
  n_ev     <- length(pred_ev)
  null_aris <- replicate(1000L, tryCatch(
    ARI(pred_ev, sample(true_ev, n_ev, replace = FALSE)),
    error = function(e) NA_real_
  ))
  p_val <- mean(null_aris >= obs_ari, na.rm = TRUE)
  cat(sprintf("   %-35s | obs=%.4f | null=%.4f | p=%.4f %s\n",
              exp$experiment, obs_ari, mean(null_aris, na.rm = TRUE), p_val,
              if (p_val < 0.05) "✓ sig" else ""))
  data.frame(experiment    = exp$experiment,
             repr          = exp$repr,
             algorithm     = exp$algorithm,
             observed_ARI  = obs_ari,
             null_mean     = round(mean(null_aris, na.rm = TRUE), 4),
             p_value       = round(p_val, 4),
             significant   = p_val < 0.05,
             stringsAsFactors = FALSE)
})
perm_df <- bind_rows(perm_rows)
sig_results$permutation <- perm_df
cat(sprintf("   %d/6 significant at p<0.05\n\n", sum(perm_df$significant)))

# --- Test 2: Chi-Squared Test of Independence (Best Model) -------------------
cat(sprintf("   Test 2: Chi-squared — best experiment (%s)...\n", best_exp_name))
best_exp  <- all_cluster_assignments[[best_exp_name]]
best_pred <- best_exp$labels
keep_best <- if (best_exp$algorithm == "hdbscan") best_pred != 0L else rep(TRUE, 7470L)
chisq_res <- chisq.test(table(cluster = best_pred[keep_best],
                              class   = true_labels[keep_best]))
cat(sprintf("   χ²=%.4f | df=%d | p=%.4e %s\n\n",
            chisq_res$statistic, chisq_res$parameter, chisq_res$p.value,
            if (chisq_res$p.value < 0.05) "✓ significant" else "not significant"))
sig_results$chisq <- list(
  experiment  = best_exp_name,
  statistic   = round(chisq_res$statistic, 4),
  df          = as.integer(chisq_res$parameter),
  p_value     = chisq_res$p.value,
  significant = chisq_res$p.value < 0.05
)

# --- Test 3: Mann-Whitney U Test (PubMedBERT vs. TF-IDF Contextual Gap) ------
cat("   Test 3: Mann-Whitney U — PubMedBERT ARI vs TF-IDF ARI...\n")
ari_tfidf  <- metrics_df$ARI[metrics_df$Representation == "tfidf"]
ari_pubmed <- metrics_df$ARI[metrics_df$Representation == "pubmedbert"]
cat(sprintf("   TF-IDF:     %s\n", paste(round(ari_tfidf,  4), collapse = ", ")))
cat(sprintf("   PubMedBERT: %s\n", paste(round(ari_pubmed, 4), collapse = ", ")))

mw_test <- wilcox.test(ari_pubmed, ari_tfidf, alternative = "greater", exact = FALSE)
n1  <- length(ari_pubmed); n2 <- length(ari_tfidf)
r_rb <- round(2 * mw_test$statistic / (n1 * n2) - 1, 4)   # rank-biserial effect size

cat(sprintf("   W=%.0f | p=%.4f (one-tailed) | r_rb=%.4f | %d/3 pubmed>tfidf %s\n",
            mw_test$statistic, mw_test$p.value, r_rb,
            sum(ari_pubmed > ari_tfidf),
            if (mw_test$p.value < 0.05) "✓ PubMedBERT > TF-IDF" else "(not significant)"))
cat("   NOTE: HDBSCAN uses different noise-exclusion subsets; comparison is not\n")
cat("         perfectly apples-to-apples. See §4.3 for full interpretation.\n\n")

sig_results$mann_whitney <- list(
  ari_tfidf         = ari_tfidf,
  ari_pubmedbert    = ari_pubmed,
  W                 = mw_test$statistic,
  p_value           = round(mw_test$p.value, 4),
  r_rb              = r_rb,
  significant       = mw_test$p.value < 0.05,
  n_pubmed_gt_tfidf = sum(ari_pubmed > ari_tfidf)
)

sig_csv <- data.frame(
  test     = c("permutation", "chisq_best", "mann_whitney_u"),
  detail   = c(paste0(sum(perm_df$significant), "/6 significant"),
               best_exp_name, "pubmedbert_vs_tfidf"),
  p_values = c(paste(perm_df$p_value, collapse = ";"),
               as.character(round(chisq_res$p.value, 6)),
               as.character(round(mw_test$p.value, 6))),
  stringsAsFactors = FALSE
)
save_artifact(sig_results, file.path(PATHS$results, "significance_tests"))
save_artifact(sig_csv,     file.path(PATHS$results, "significance_summary"))

# =============================================================================
# Section 5: Direct Answers to Research Questions
# Summarizes empirical evidence addressing the core hypotheses of the paper.
# =============================================================================
cat("── RQ ANSWERS ──\n")
best_tfidf <- metrics_df[metrics_df$Representation == "tfidf", ]
rq1        <- best_tfidf[which.max(best_tfidf$ARI), ]
cat(sprintf("RQ1 (best TF-IDF algorithm): %s | ARI=%.4f | MacroF1=%.4f\n",
            toupper(rq1$Algorithm), rq1$ARI, rq1$Macro_F1))
cat(sprintf("RQ2 (PubMedBERT vs TF-IDF): mean ARI %.4f → %.4f | %d/3 algorithms PubMedBERT wins\n",
            mean(ari_tfidf), mean(ari_pubmed), sum(ari_pubmed > ari_tfidf)))
cat(sprintf("     Mann-Whitney p=%.4f %s\n",
            mw_test$p.value,
            if (mw_test$p.value < 0.05) "→ significant" else "→ not significant (HDBSCAN inflates TF-IDF ARI via noise exclusion)"))
cat(sprintf("Best overall: %s + %s | ARI=%.4f | MacroF1=%.4f\n",
            REPR_DISPLAY[best_row$Representation], ALG_DISPLAY[best_row$Algorithm],
            best_row$ARI, best_row$Macro_F1))

cat(sprintf("\n── 05_evaluation.R complete (%.1f sec)\n", elapsed(t0)))
cat("   results/table3.csv | results/per_class_f1.csv\n")
cat("   results/significance_tests.rds | results/significance_summary.csv\n")
cat("   Next: source('R/06_keywords.R')\n")