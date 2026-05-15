# =============================================================================
# 03_umap_pca.R — Dimensionality Reduction: PCA (2D) + UMAP (50D + 2D)
# BioCluster-R | ICCA 2026
#
# Inputs  : data/repr/repr_tfidf.rds
#           data/repr/repr_pubmedbert.rds
#           data/processed/corpus_balanced.rds
#
# Outputs : data/umap/pca2_tfidf.rds          ← 7470 × 2  (Figure 2 ONLY)
#           data/umap/pca2_pubmedbert.rds      ← 7470 × 2  (Figure 2 ONLY)
#           data/umap/umap50_tfidf.rds         ← 7470 × 50 (clustering input)
#           data/umap/umap50_pubmedbert.rds    ← 7470 × 50 (clustering input)
#           data/umap/umap2_tfidf.rds          ← 7470 × 2  (Figure 2 + Figure 4)
#           data/umap/umap2_pubmedbert.rds     ← 7470 × 2  (Figure 2 + Figure 4)
#           data/umap/umap_summary.csv/.rds
#
# CRITICAL: PCA outputs are NEVER passed to any clustering algorithm (§3.5).
#           They exist solely to demonstrate that linear DR fails to separate
#           disease classes (Figure 2), motivating the non-linear UMAP choice.
#
# Run time: ~20 min (TF-IDF UMAP dominates due to ~10K–30K input dimensions)
# Next    : 04_clustering.R
# =============================================================================

source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(uwot)
  library(irlba)
  library(Matrix)
})

set.seed(42)
t0 <- proc.time()
cat("── 03_umap_pca.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# =============================================================================
# Section 0: Data Loading and Alignment Validation
# Loads TF-IDF and PubMedBERT representations along with the balanced corpus.
# Verifies precise document ID alignment to ensure consistency for DR steps.
# =============================================================================
cat("0. Loading representations and corpus...\n")

corpus <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))
repr_tfidf <- readRDS(file.path(PATHS$repr, "repr_tfidf.rds"))
repr_pubmedbert <- readRDS(file.path(PATHS$repr, "repr_pubmedbert.rds"))

corpus_ids <- corpus$doc_id # authoritative row order: doc1 … doc7470

stopifnot(
  "corpus row count wrong"       = nrow(corpus) == 7470,
  "tfidf row count wrong"        = nrow(repr_tfidf) == 7470,
  "pubmedbert row count wrong"   = nrow(repr_pubmedbert) == 7470,
  "pubmedbert dim count wrong"   = ncol(repr_pubmedbert) == 768,
  "tfidf doc_id misaligned"      = all(rownames(repr_tfidf) == corpus_ids),
  "pubmedbert doc_id misaligned" = all(rownames(repr_pubmedbert) == corpus_ids)
)

cat(sprintf("   corpus      : %d docs\n", nrow(corpus)))
cat(sprintf("   repr_tfidf  : %d × %d (sparse)\n", nrow(repr_tfidf), ncol(repr_tfidf)))
cat(sprintf("   repr_pubmed : %d × %d (dense)\n", nrow(repr_pubmedbert), ncol(repr_pubmedbert)))
cat("   All alignment checks passed ✓\n\n")

# Named list of representations — iterated in both PCA and UMAP sections
reprs <- list(
  tfidf      = repr_tfidf,
  pubmedbert = repr_pubmedbert
)

# =============================================================================
# Section 1: Principal Component Analysis (PCA) Baseline
# Computes 2D linear projections exclusively for Figure 2 comparison.
# This establishes the baseline performance limitation of linear transformations
# on this dataset, motivating the necessity of non-linear DR methods (UMAP).
# =============================================================================
cat("1. PCA — 2D linear baseline (for Figure 2 only)...\n")
t_pca <- proc.time()

# TF-IDF PCA: truncated SVD — irlba::irlba(nv=2) as specified in §3.5
svd_tfidf <- irlba::irlba(repr_tfidf, nv = 2, nu = 0, verbose = FALSE)
# Projection onto top-2 right singular vectors: X %*% V
pca2_tfidf <- as.matrix(repr_tfidf %*% svd_tfidf$v)
rownames(pca2_tfidf) <- corpus_ids
colnames(pca2_tfidf) <- c("PC1", "PC2")

cat(sprintf(
  "   pca2_tfidf     : %d × %d | range [%.3f, %.3f]\n",
  nrow(pca2_tfidf), ncol(pca2_tfidf),
  min(pca2_tfidf), max(pca2_tfidf)
))

# PubMedBERT PCA: standard prcomp — prcomp(rank.=2) as specified in §3.5
pca_pubmed <- prcomp(repr_pubmedbert, rank. = 2, center = TRUE, scale. = FALSE)
pca2_pubmedbert <- pca_pubmed$x # 7470 × 2 scores matrix
rownames(pca2_pubmedbert) <- corpus_ids
colnames(pca2_pubmedbert) <- c("PC1", "PC2")

cat(sprintf(
  "   pca2_pubmedbert: %d × %d | range [%.3f, %.3f]\n",
  nrow(pca2_pubmedbert), ncol(pca2_pubmedbert),
  min(pca2_pubmedbert), max(pca2_pubmedbert)
))

saveRDS(pca2_tfidf, file.path(PATHS$umap, "pca2_tfidf.rds"))
saveRDS(pca2_pubmedbert, file.path(PATHS$umap, "pca2_pubmedbert.rds"))
cat(sprintf("   PCA elapsed: %.1f sec\n\n", elapsed(t_pca)))

# =============================================================================
# Section 2: Shared UMAP Function with Parameter Specifications
# Standardizes UMAP execution across dimensions (2D for viz, 50D for clustering).
# Enforces exact reproducibility by restricting computation to a single thread
# and applying strict random seed initialization per execution.
# =============================================================================

# Helper: run UMAP, validate output, attach rownames
run_umap <- function(mat, n_components, min_dist, repr_name) {
  tag <- sprintf("%s %dD", repr_name, n_components)
  cat(sprintf(
    "   Running UMAP: %s | input %d × %d → output %d dims...\n",
    tag, nrow(mat), ncol(mat), n_components
  ))
  t_u <- proc.time()

  set.seed(42) # belt-and-suspenders: R seed before uwot call
  emb <- uwot::umap(
    X            = as.matrix(mat), # as.matrix() handles sparse input safely
    n_components = n_components,
    n_neighbors  = 30L,
    min_dist     = min_dist,
    metric       = "cosine",
    n_threads    = 1L, # MANDATORY for bit-identical reproducibility
    seed         = 42L,
    verbose      = FALSE
  )

  # Validate output
  stopifnot(
    "UMAP row count wrong" = nrow(emb) == 7470,
    "UMAP col count wrong" = ncol(emb) == n_components,
    "UMAP contains NaN"    = !any(is.nan(emb)),
    "UMAP contains Inf"    = !any(is.infinite(emb))
  )

  rownames(emb) <- corpus_ids
  colnames(emb) <- paste0("UMAP", seq_len(n_components))

  cat(sprintf(
    "   %s complete: %.1f sec | range [%.3f, %.3f]\n",
    tag, elapsed(t_u), min(emb), max(emb)
  ))
  emb
}

# =============================================================================
# Section 3: UMAP 50-Dimensional Projections (Clustering Input)
# Reduces dimensionality to 50 to mitigate the curse of dimensionality, preserving
# local manifold structures while optimizing density separation for clustering.
# Uses min_dist=0.0 to strictly maximize intra-cluster density.
# =============================================================================
cat("2. UMAP 50D — clustering input (~15–20 min for TF-IDF)...\n")
t_umap50 <- proc.time()

umap50_tfidf <- run_umap(repr_tfidf, n_components = 50L, min_dist = 0.0, repr_name = "tfidf")
umap50_pubmedbert <- run_umap(repr_pubmedbert, n_components = 50L, min_dist = 0.0, repr_name = "pubmedbert")

saveRDS(umap50_tfidf, file.path(PATHS$umap, "umap50_tfidf.rds"))
saveRDS(umap50_pubmedbert, file.path(PATHS$umap, "umap50_pubmedbert.rds"))

cat(sprintf("   UMAP 50D both saved | elapsed: %.1f sec\n\n", elapsed(t_umap50)))

# =============================================================================
# Section 4: UMAP 2-Dimensional Projections (Visualization)
# Generates 2D coordinates for visual cluster inspection (Figures 2 and 4).
# Employs min_dist=0.1 to avoid complete overplotting and ensure visibility.
# These projections are explicitly excluded from quantitative clustering.
# =============================================================================
cat("3. UMAP 2D — visualisation (~2–4 min)...\n")
t_umap2 <- proc.time()

umap2_tfidf <- run_umap(repr_tfidf, n_components = 2L, min_dist = 0.1, repr_name = "tfidf")
umap2_pubmedbert <- run_umap(repr_pubmedbert, n_components = 2L, min_dist = 0.1, repr_name = "pubmedbert")

saveRDS(umap2_tfidf, file.path(PATHS$umap, "umap2_tfidf.rds"))
saveRDS(umap2_pubmedbert, file.path(PATHS$umap, "umap2_pubmedbert.rds"))

cat(sprintf("   UMAP 2D both saved | elapsed: %.1f sec\n\n", elapsed(t_umap2)))

# =============================================================================
# Section 5: Matrix Alignment Quality Assurance
# Validates that all generated PCA and UMAP matrices perfectly align with
# the authoritative document index prior to serialization.
# =============================================================================
cat("4. Alignment verification...\n")

all_outputs <- list(
  pca2_tfidf      = pca2_tfidf,
  pca2_pubmedbert = pca2_pubmedbert,
  umap50_tfidf    = umap50_tfidf,
  umap50_pubmed   = umap50_pubmedbert,
  umap2_tfidf     = umap2_tfidf,
  umap2_pubmed    = umap2_pubmedbert
)

for (name in names(all_outputs)) {
  m <- all_outputs[[name]]
  ok <- all(rownames(m) == corpus_ids)
  cat(sprintf(
    "   %-20s : %d × %-3d | doc_id aligned: %s\n",
    name, nrow(m), ncol(m),
    if (ok) "✓" else "FAIL"
  ))
  if (!ok) stop("doc_id alignment failed for: ", name)
}

# =============================================================================
# Section 6: Provenance and Output Serialization
# Catalogs the generated dimensionality reduction matrices and persists
# them to disk alongside a summary tracking matrix metadata.
# =============================================================================
summary_tbl <- tibble(
  output = names(all_outputs),
  rows = sapply(all_outputs, nrow),
  dims = sapply(all_outputs, ncol),
  used_for = c(
    "Figure 2 (PCA vs UMAP panel)",
    "Figure 2 (PCA vs UMAP panel)",
    "04_clustering.R input",
    "04_clustering.R input",
    "Figure 2 + Figure 4",
    "Figure 2 + Figure 4"
  ),
  file = c(
    "pca2_tfidf.rds",
    "pca2_pubmedbert.rds",
    "umap50_tfidf.rds",
    "umap50_pubmedbert.rds",
    "umap2_tfidf.rds",
    "umap2_pubmedbert.rds"
  )
)

print(summary_tbl)
save_artifact(summary_tbl, file.path(PATHS$umap, "umap_summary"))

cat(sprintf("\n── 03_umap_pca.R complete (%.1f sec)\n", elapsed(t0)))
cat("   Saved to data/umap/:\n")
cat("     pca2_tfidf.rds, pca2_pubmedbert.rds\n")
cat("     umap50_tfidf.rds, umap50_pubmedbert.rds  ← clustering inputs\n")
cat("     umap2_tfidf.rds, umap2_pubmedbert.rds    ← visualisation only\n")
cat("   Next: source('R/04_clustering.R')\n")
