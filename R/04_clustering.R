# =============================================================================
# 04_clustering.R — K-Means, HDBSCAN, Agglomerative Clustering
# BioCluster-R | ICCA 2026
#
# Inputs  : data/processed/corpus_balanced.rds
#           data/umap/umap50_tfidf.rds
#           data/umap/umap50_pubmedbert.rds
#
# Outputs : data/clustering/sil_subsample_idx.rds
#           data/clustering/kmeans_sweep_{repr}.rds/.csv
#           data/clustering/labels_kmeans_{repr}.rds
#           data/clustering/kmeans_meta.rds/.csv
#           data/clustering/hdbscan_grid_{repr}.rds/.csv
#           data/clustering/labels_hdbscan_{repr}.rds
#           data/clustering/hdbscan_meta.rds/.csv
#           data/clustering/hclust_{repr}.rds
#           data/clustering/labels_agg_{repr}.rds
#           data/clustering/agg_meta.rds/.csv
#           data/clustering/all_cluster_assignments.rds
#
# ALGORITHMIC EXPERIMENTS:
#   K-Means      : Optimal k selected via silhouette score maximization (k=2:10).
#   HDBSCAN      : Density parameter optimized for ~30% noise tolerance (k ≤ 20).
#   Agglomerative: Number of clusters fixed to k=5 using Ward.D2 linkage.
#
# Next    : 05_evaluation.R
# =============================================================================

source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(cluster)
  library(dbscan)
  library(fastcluster)
})

set.seed(42)
t0 <- proc.time()
cat("── 04_clustering.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# =============================================================================
# Section 0: Loading Preprocessed Dimensionality Data
# Imports the 50D UMAP projections for both TF-IDF and PubMedBERT embeddings.
# =============================================================================
cat("0. Loading inputs...\n")
corpus <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))
stopifnot(nrow(corpus) == 7470)
true_labels <- corpus$label
corpus_ids  <- corpus$doc_id

umap50 <- list(
  tfidf      = readRDS(file.path(PATHS$umap, "umap50_tfidf.rds")),
  pubmedbert = readRDS(file.path(PATHS$umap, "umap50_pubmedbert.rds"))
)
for (nm in names(umap50)) {
  stopifnot(nrow(umap50[[nm]]) == 7470, ncol(umap50[[nm]]) == 50,
            all(rownames(umap50[[nm]]) == corpus_ids))
  cat(sprintf("   umap50_%s: %d × %d ✓\n", nm, nrow(umap50[[nm]]), ncol(umap50[[nm]])))
}

# =============================================================================
# Section 1: Stratified Subsampling for Evaluation Efficiency
# Isolates a balanced subset of 2,000 documents (400 per disease class) to ensure 
# computationally feasible and unbiased silhouette score calculations.
# =============================================================================
cat("\n1. Stratified silhouette subsample (400 per class)...\n")
set.seed(42)
sil_idx <- unlist(lapply(1:5, function(lbl) sample(which(corpus$label == lbl), 400L)))
stopifnot(length(sil_idx) == 2000L)
saveRDS(sil_idx, file.path(PATHS$clustering, "sil_subsample_idx.rds"))
cat(sprintf("   %d docs | 400 per class ✓\n\n", length(sil_idx)))

kmeans_meta_rows  <- list()
hdbscan_meta_rows <- list()
agg_meta_rows     <- list()
all_labels        <- list()

# =============================================================================
# Section 2: K-Means Clustering Algorithm Pipeline
# Executes a silhouette-guided parameter sweep to automatically identify the 
# optimal number of clusters (k) without enforcing ground-truth expectations.
# =============================================================================
cat("2. K-Means clustering...\n")
KMEANS_K_RANGE <- 2:10

for (repr_name in names(umap50)) {
  cat(sprintf("\n   ── K-Means: %s ──\n", repr_name))
  mat      <- as.matrix(umap50[[repr_name]])
  sil_mat  <- mat[sil_idx, ]
  sil_dist <- dist(sil_mat)

  sweep_rows <- lapply(KMEANS_K_RANGE, function(k) {
    set.seed(42)
    km <- kmeans(mat, centers = k, nstart = 20L,
                 iter.max = 300L, algorithm = "Hartigan-Wong")
    sil_score <- if (length(unique(km$cluster[sil_idx])) >= 2L)
      mean(silhouette(km$cluster[sil_idx], sil_dist)[, 3]) else NA_real_
    cat(sprintf("   k=%2d | sil=%.4f\n", k, sil_score))
    data.frame(k = k, silhouette = sil_score, wss = km$tot.withinss)
  })
  sweep_df      <- bind_rows(sweep_rows)
  sweep_df$repr <- repr_name
  save_artifact(sweep_df, file.path(PATHS$clustering, paste0("kmeans_sweep_", repr_name)))

  best_k <- KMEANS_K_RANGE[which.max(sweep_df$silhouette)]
  cat(sprintf("   Selected k=%d (sil=%.4f)%s\n",
              best_k, max(sweep_df$silhouette, na.rm = TRUE),
              if (best_k == 5L) " ← matches ground-truth k=5 ✓" else " ← WARNING: ≠ k=5"))

  set.seed(42)
  km_final  <- kmeans(mat, centers = best_k, nstart = 50L,
                      iter.max = 300L, algorithm = "Hartigan-Wong")
  labels_km <- km_final$cluster
  names(labels_km) <- corpus_ids

  sil_final <- mean(silhouette(labels_km[sil_idx], sil_dist)[, 3])
  cat(sprintf("   Final: k=%d | iter=%d | sil=%.4f | WSS=%.1f\n",
              best_k, km_final$iter, sil_final, km_final$tot.withinss))
  cat(sprintf("   Sizes: %s\n",
              paste(sort(table(labels_km), decreasing = TRUE), collapse = " | ")))

  saveRDS(labels_km, file.path(PATHS$clustering, paste0("labels_kmeans_", repr_name, ".rds")))
  all_labels[[paste0(repr_name, "_kmeans")]] <- labels_km
  kmeans_meta_rows[[repr_name]] <- data.frame(
    repr = repr_name, selected_k = best_k, matches_gt_k = best_k == 5L,
    silhouette = round(sil_final, 4), wss = round(km_final$tot.withinss, 2),
    n_iter = km_final$iter, stringsAsFactors = FALSE
  )
}

save_artifact(bind_rows(kmeans_meta_rows), file.path(PATHS$clustering, "kmeans_meta"))
cat("\nK-Means complete ✓\n\n")

# =============================================================================
# Section 3: HDBSCAN Clustering Algorithm Pipeline
# Performs density-based spatial clustering. Optimizes the 'minPts' hyperparameter 
# to balance noise exclusion (~30%) against meaningful cluster formation (k ≤ 20).
# Restricts highly degenerate clustering topologies to ensure interpretability.
# =============================================================================
cat("3. HDBSCAN clustering...\n")
MINPTS_GRID <- c(10L, 25L, 50L, 100L, 200L, 500L)
HDBSCAN_MAX_K <- 20L   # configs with k > 20 excluded from selection

for (repr_name in names(umap50)) {
  cat(sprintf("\n   ── HDBSCAN: %s ──\n", repr_name))
  mat <- as.matrix(umap50[[repr_name]])

  grid_rows <- lapply(MINPTS_GRID, function(mp) {
    hdb       <- hdbscan(mat, minPts = mp)
    labs      <- hdb$cluster
    n_noise   <- sum(labs == 0L)
    noise_pct <- round(100 * n_noise / 7470, 1)
    n_cls     <- length(unique(labs[labs != 0L]))
    degen     <- if (n_cls > 0L)
      max(table(labs[labs != 0L])) / sum(labs != 0L) > 0.80 else TRUE
    cat(sprintf("   minPts=%-3d | k=%-4d | noise=%.1f%% | degen=%s%s\n",
                mp, n_cls, noise_pct, degen,
                if (n_cls > HDBSCAN_MAX_K) " [k>20 excluded]" else ""))
    data.frame(repr = repr_name, minPts = mp, n_clusters = n_cls,
               n_noise = n_noise, noise_pct = noise_pct,
               degenerate = degen, stringsAsFactors = FALSE)
  })
  grid_df <- bind_rows(grid_rows)
  save_artifact(grid_df, file.path(PATHS$clustering, paste0("hdbscan_grid_", repr_name)))

  # Valid: non-degenerate AND k ≤ 20
  valid <- grid_df[!grid_df$degenerate & grid_df$n_clusters <= HDBSCAN_MAX_K, ]

  if (nrow(valid) > 0L) {
    selected_row <- valid[which.min(abs(valid$noise_pct - 30)), ]
  } else {
    # Fallback: any non-degenerate (even k > 20), then minimum noise
    warning(sprintf("HDBSCAN %s: no non-degenerate k≤20 found. Fallback to highest minPts.", repr_name))
    non_degen <- grid_df[!grid_df$degenerate, ]
    selected_row <- if (nrow(non_degen) > 0L)
      non_degen[which.min(abs(non_degen$noise_pct - 30)), ]
    else grid_df[nrow(grid_df), ]
  }

  selected_mp <- selected_row$minPts
  cat(sprintf("   Selected minPts=%d (k=%d, noise=%.1f%%)%s\n",
              selected_mp, selected_row$n_clusters, selected_row$noise_pct,
              if (selected_row$n_clusters > HDBSCAN_MAX_K) " [fallback — k>20]" else ""))

  hdb_final  <- hdbscan(mat, minPts = selected_mp)
  labels_hdb <- hdb_final$cluster
  names(labels_hdb) <- corpus_ids

  n_noise   <- sum(labels_hdb == 0L)
  noise_pct <- round(100 * n_noise / 7470, 1)
  n_cls     <- length(unique(labels_hdb[labels_hdb != 0L]))

  sil_eval <- intersect(sil_idx, which(labels_hdb != 0L))
  sil_hdb  <- if (length(unique(labels_hdb[sil_eval])) >= 2L && length(sil_eval) >= 10L)
    mean(silhouette(labels_hdb[sil_eval], dist(mat[sil_eval, ]))[, 3]) else NA_real_

  cat(sprintf("   Final: k=%d | noise=%d (%.1f%%) | sil=%.4f\n",
              n_cls, n_noise, noise_pct, sil_hdb))
  cat(sprintf("   Sizes: %s%s\n",
              paste(head(sort(table(labels_hdb[labels_hdb != 0L]), decreasing=TRUE), 10),
                    collapse=" | "),
              if (n_cls > 10L) sprintf(" ... (%d total)", n_cls) else ""))

  saveRDS(labels_hdb, file.path(PATHS$clustering, paste0("labels_hdbscan_", repr_name, ".rds")))
  all_labels[[paste0(repr_name, "_hdbscan")]] <- labels_hdb
  hdbscan_meta_rows[[repr_name]] <- data.frame(
    repr = repr_name, selected_minPts = selected_mp, k_found = n_cls,
    n_noise = n_noise, noise_pct = noise_pct,
    k_exceeds_20 = n_cls > HDBSCAN_MAX_K,
    silhouette = round(sil_hdb, 4), stringsAsFactors = FALSE
  )
}

save_artifact(bind_rows(hdbscan_meta_rows), file.path(PATHS$clustering, "hdbscan_meta"))
cat("\nHDBSCAN complete ✓\n\n")

# =============================================================================
# Section 4: Agglomerative Hierarchical Clustering Pipeline
# Constructs a complete linkage hierarchy using Ward's minimum variance method.
# Enforces k=5 due to empirical limitations of gap statistic evaluations on this 
# topology. Diagnostic k is retained for transparency.
# =============================================================================
cat("4. Agglomerative clustering...\n")

for (repr_name in names(umap50)) {
  cat(sprintf("\n   ── Agglomerative: %s ──\n", repr_name))
  mat <- as.matrix(umap50[[repr_name]])

  t_dist <- proc.time()
  dmat   <- dist(mat, method = "euclidean")
  cat(sprintf("   Distance matrix: %.1f sec\n", elapsed(t_dist)))

  hc <- fastcluster::hclust(dmat, method = "ward.D2")
  saveRDS(hc, file.path(PATHS$clustering, paste0("hclust_", repr_name, ".rds")))

  # Diagnostic: gap-selected k (not used for Table 3)
  heights_desc <- rev(hc$height)
  gap_k_raw    <- which.max(abs(diff(heights_desc))) + 1L
  gap_k        <- max(2L, min(10L, as.integer(gap_k_raw)))
  cat(sprintf("   Gap method → k=%d (diagnostic only)\n", gap_k))

  # Primary: forced k=5
  labels_agg <- cutree(hc, k = 5L)
  names(labels_agg) <- corpus_ids

  sil_mat  <- mat[sil_idx, ]
  sil_dist <- dist(sil_mat)
  sil_agg  <- mean(silhouette(labels_agg[sil_idx], sil_dist)[, 3])

  cat(sprintf("   Forced k=5 | sil=%.4f\n", sil_agg))
  cat(sprintf("   Sizes: %s\n",
              paste(sort(table(labels_agg), decreasing = TRUE), collapse = " | ")))

  saveRDS(labels_agg, file.path(PATHS$clustering, paste0("labels_agg_", repr_name, ".rds")))
  all_labels[[paste0(repr_name, "_agglomerative")]] <- labels_agg
  agg_meta_rows[[repr_name]] <- data.frame(
    repr = repr_name, primary_k = 5L, gap_k = gap_k, gap_k_raw = as.integer(gap_k_raw),
    silhouette_k5 = round(sil_agg, 4), height_max = round(max(hc$height), 4),
    stringsAsFactors = FALSE
  )
}

save_artifact(bind_rows(agg_meta_rows), file.path(PATHS$clustering, "agg_meta"))
cat("\nAgglomerative complete ✓\n\n")

# =============================================================================
# Section 5: Aggregation and Serialization of Results
# Consolidates the outputs of all six experimental configurations into a single 
# standardized artifact for downstream evaluation and figure generation.
# =============================================================================
cat("5. Assembling all_cluster_assignments.rds...\n")

all_cluster_assignments <- lapply(names(all_labels), function(exp_name) {
  parts <- strsplit(exp_name, "_")[[1]]
  list(experiment = exp_name, repr = parts[1],
       algorithm  = paste(parts[-1], collapse = "_"),
       labels = all_labels[[exp_name]], doc_ids = corpus_ids)
})
names(all_cluster_assignments) <- names(all_labels)
stopifnot("Expected 6 experiments" = length(all_cluster_assignments) == 6L)

for (i in seq_along(all_cluster_assignments)) {
  x <- all_cluster_assignments[[i]]
  cat(sprintf("   [%d] %-35s | k=%-4d | noise=%d\n",
              i, x$experiment,
              length(unique(x$labels[x$labels != 0L])),
              sum(x$labels == 0L)))
}

saveRDS(all_cluster_assignments,
        file.path(PATHS$clustering, "all_cluster_assignments.rds"))

cat(sprintf("\n── 04_clustering.R complete (%.1f sec)\n", elapsed(t0)))
cat("   K-Means      : data-driven k (silhouette sweep)\n")
cat("   HDBSCAN      : noise-to-30%, constrained k ≤ 20\n")
cat("   Agglomerative: forced k=5 (gap_k saved in agg_meta.csv)\n")
cat("   Next: source('R/05_evaluation.R')\n")
