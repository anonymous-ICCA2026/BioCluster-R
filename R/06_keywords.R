# =============================================================================
# 06_keywords.R — c-TF-IDF Keyword Extraction & Disease Knowledge Mapping
# BioCluster-R | ICCA 2026
#
# Inputs  : data/processed/corpus_balanced.rds
#           data/repr/vocab_tfidf.rds
#           data/clustering/all_cluster_assignments.rds
#           results/table3.rds
#           results/per_class_f1.rds
#
# Outputs : results/table4.rds/.csv             ← Table 4 (paper)
#           results/cluster_keywords_full.rds/.csv
#           results/cluster_keywords_wide.rds/.csv
#           results/noise_keywords.rds/.csv      ← HDBSCAN noise profile
#
# Method  : c-TF-IDF (§3.9) — each cluster's docs concatenated into one
#           super-document; TF-IDF across k super-documents identifies
#           cluster-distinctive terms. Vocabulary = vocab_tfidf.rds.
#           Hungarian assignment maps cluster → disease class.
#           F1 lookup uses class integer (not disease_name string) to avoid
#           any column naming issues from 05_evaluation.R's per_class_f1.
#
# Next    : 07_figures.R
# =============================================================================

source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(text2vec)
  library(Matrix)
  library(clue)
})

set.seed(42)
t0 <- proc.time()
cat("── 06_keywords.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# =============================================================================
# Section 0: Loading Dependencies and Evaluation Artifacts
# Imports vocabulary arrays and the optimal clustering topology determined 
# during the empirical evaluation phase.
# =============================================================================
cat("0. Loading inputs...\n")
corpus       <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))
vocab_tfidf  <- readRDS(file.path(PATHS$repr,      "vocab_tfidf.rds"))
all_cluster  <- readRDS(file.path(PATHS$clustering, "all_cluster_assignments.rds"))
table3       <- readRDS(file.path(PATHS$results,   "table3.rds"))
per_class_f1 <- readRDS(file.path(PATHS$results,   "per_class_f1.rds"))

true_labels <- corpus$label
corpus_ids  <- corpus$doc_id

# Reverse map: disease_name → class integer (for F1 lookup by class)
disease_to_class <- setNames(1:5, DISEASE_MAP[as.character(1:5)])

cat(sprintf("   corpus: %d docs | vocab: %d terms\n",
            nrow(corpus), nrow(vocab_tfidf)))
cat(sprintf("   per_class_f1: %d rows | columns: %s\n",
            nrow(per_class_f1), paste(names(per_class_f1), collapse=", ")))

# =============================================================================
# Section 1: Selection of the Optimal Representation Pipeline
# Automatically retrieves the configuration demonstrating the highest ARI to 
# serve as the baseline for thematic cluster profiling.
# =============================================================================
cat("\n1. Identifying best experiment...\n")

best_row <- table3[which.max(table3$ARI), ]

# Convert display names back to experiment key
best_repr <- tolower(best_row$Representation) %>%
  str_replace("tf-idf", "tfidf") %>%
  str_replace("pubmedbert", "pubmedbert")
best_alg  <- tolower(best_row$Algorithm) %>%
  str_replace("k-means", "kmeans") %>%
  str_replace("agglomerative", "agglomerative") %>%
  str_replace("hdbscan", "hdbscan")
best_exp_name <- paste0(best_repr, "_", best_alg)

# Verify
if (!best_exp_name %in% names(all_cluster)) {
  stop("Could not resolve best experiment name: '", best_exp_name, "'\n",
       "Available: ", paste(names(all_cluster), collapse=", "))
}

best_exp    <- all_cluster[[best_exp_name]]
labels_best <- best_exp$labels
is_hdbscan  <- best_exp$algorithm == "hdbscan"
k_eff       <- length(unique(labels_best[labels_best != 0L]))
n_noise     <- sum(labels_best == 0L)

cat(sprintf("   Best experiment: %s\n", best_exp_name))
cat(sprintf("   ARI=%.4f | k=%d | noise=%d docs (%.1f%%)\n",
            best_row$ARI, k_eff, n_noise, 100 * n_noise / 7470))

# =============================================================================
# Section 2: Optimal Cluster-to-Class Alignment
# Employs the Hungarian algorithm to definitively map empirically derived 
# clusters to ground-truth disease classes, enabling direct interpretability.
# =============================================================================
cat("\n2. Computing Hungarian cluster-to-class assignment...\n")

pred_nn <- labels_best[labels_best != 0L]
true_nn <- true_labels[labels_best != 0L]
classes  <- 1:5
clusters <- sort(unique(pred_nn))
n_cls    <- length(classes)
n_clu    <- length(clusters)

ct_assign <- matrix(0L, nrow = n_cls, ncol = max(n_cls, n_clu))
for (i in seq_along(classes))
  for (j in seq_along(clusters))
    ct_assign[i, j] <- sum(true_nn == classes[i] & pred_nn == clusters[j])

assignment <- clue::solve_LSAP(ct_assign, maximum = TRUE)

# Establish definitive mappings between derived clusters and biomedical ontologies
cluster_disease_map <- character(0)
class_for_cluster   <- integer(0)
for (i in seq_along(classes)) {
  j <- assignment[i]
  if (j <= n_clu) {
    cl_char <- as.character(clusters[j])
    cluster_disease_map[cl_char] <- DISEASE_MAP[as.character(classes[i])]
    class_for_cluster[cl_char]   <- classes[i]
  }
}

cat("   Cluster → Disease:\n")
for (cl in sort(clusters)) {
  cl_char <- as.character(cl)
  dis     <- if (cl_char %in% names(cluster_disease_map))
    cluster_disease_map[cl_char] else "Unassigned (sub-cluster)"
  cat(sprintf("   Cluster %d → %s\n", cl, dis))
}

# =============================================================================
# Section 3: Class-Based TF-IDF (c-TF-IDF) Keyword Extraction
# Treats each cluster as a singular 'super-document' to extract statistically 
# distinctive biomedical terms unique to that specific disease subgroup.
# =============================================================================
cat("\n3. Computing c-TF-IDF...\n")

vectorizer <- vocab_vectorizer(vocab_tfidf)

cluster_texts <- sapply(clusters, function(cl) {
  paste(corpus$text_clean[labels_best == cl], collapse = " ")
})
names(cluster_texts) <- as.character(clusters)

it_clusters  <- itoken(cluster_texts, tokenizer = word_tokenizer,
                       ids = as.character(clusters), progressbar = FALSE)
dtm_clusters <- create_dtm(it_clusters, vectorizer)
cat(sprintf("   Super-doc DTM: %d × %d\n", nrow(dtm_clusters), ncol(dtm_clusters)))

tfidf_model  <- TfIdf$new()
dtm_ctfidf   <- fit_transform(dtm_clusters, tfidf_model)
term_names   <- colnames(dtm_ctfidf)

# =============================================================================
# Section 4: Thematic Profiling and Lexical Evaluation
# Isolates the top 10 discriminative keywords per subgroup and cross-references 
# them with previously computed per-class F1 metrics for unified analysis.
# =============================================================================
cat("\n4. Extracting top 10 keywords per cluster...\n\n")

keyword_rows <- lapply(seq_along(clusters), function(i) {
  cl      <- clusters[i]
  cl_char <- as.character(cl)
  scores  <- as.numeric(dtm_ctfidf[i, ])
  top10   <- term_names[order(scores, decreasing = TRUE)[1:10]]
  
  dis_name  <- if (cl_char %in% names(cluster_disease_map))
    cluster_disease_map[cl_char] else "Unassigned"
  dis_class <- if (cl_char %in% names(class_for_cluster))
    class_for_cluster[cl_char] else NA_integer_
  
  n_docs  <- sum(labels_best == cl)
  ct_cl   <- table(true_labels[labels_best == cl])
  purity  <- round(max(ct_cl) / n_docs, 4)
  
  # Retrieve precise F1 metrics via deterministic integer mapping
  f1_val <- if (!is.na(dis_class)) {
    f1_row <- per_class_f1[per_class_f1$experiment == best_exp_name &
                             per_class_f1$class    == dis_class, ]
    if (nrow(f1_row) > 0) round(f1_row$f1[1], 4) else NA_real_
  } else { NA_real_ }
  
  cat(sprintf("   Cluster %d [%-35s | n=%4d | purity=%.3f | F1=%.3f]:\n   %s\n",
              cl, dis_name, n_docs, purity,
              ifelse(is.na(f1_val), 0, f1_val),
              paste(top10, collapse = ", ")))
  
  df <- data.frame(
    cluster          = cl,
    assigned_disease = dis_name,
    assigned_class   = dis_class,
    n_docs           = n_docs,
    purity           = purity,
    f1               = f1_val,
    keywords         = paste(top10, collapse = ", "),
    stringsAsFactors = FALSE
  )
  for (ki in 1:10) df[[paste0("kw", ki)]] <- top10[ki]
  df
})
keyword_df <- bind_rows(keyword_rows)

# Full keyword-score table
full_keywords <- lapply(seq_along(clusters), function(i) {
  cl     <- clusters[i]
  scores <- as.numeric(dtm_ctfidf[i, ])
  keep   <- scores > 0
  data.frame(cluster = cl, term = term_names[keep],
             ctfidf_score = round(scores[keep], 6),
             stringsAsFactors = FALSE) %>%
    arrange(desc(ctfidf_score))
})
full_keyword_df <- bind_rows(full_keywords)

# =============================================================================
# Section 5: Linguistic Profiling of the Noise Distribution
# If utilizing density-based algorithms, analyzes the aggregate noise 
# distribution to understand what biomedical terms resist distinct clustering.
# =============================================================================
noise_df <- NULL
if (is_hdbscan && n_noise > 0L) {
  cat(sprintf("\n5. Noise keyword profile (%d documents)...\n", n_noise))
  
  # Construct a unified textual representation of unclustered noise documents
  noise_text <- paste(corpus$text_clean[labels_best == 0L], collapse = " ")
  it_noise   <- itoken(noise_text, tokenizer = word_tokenizer,
                       ids = "noise", progressbar = FALSE)
  dtm_noise  <- create_dtm(it_noise, vectorizer)
  
  if (nrow(dtm_noise) > 0L) {
    scores_noise <- as.numeric(dtm_noise[1, ])
    top10_noise  <- term_names[order(scores_noise, decreasing = TRUE)[1:10]]
    cat(sprintf("   Noise top terms: %s\n", paste(top10_noise, collapse = ", ")))
  } else {
    # Implement term frequency fallback for edge cases with empty vectorization
    cat("   WARNING: dtm_noise was empty. Using raw frequency fallback.\n")
    noise_tokens <- unlist(strsplit(noise_text, "\\s+"))
    noise_freq   <- sort(table(noise_tokens[noise_tokens %in% term_names]),
                         decreasing = TRUE)
    top10_noise  <- names(head(noise_freq, 10))
    cat(sprintf("   Noise top terms (freq fallback): %s\n",
                paste(top10_noise, collapse = ", ")))
  }
  
  noise_df <- data.frame(
    cluster = 0L, assigned_disease = "Noise (unassigned)",
    assigned_class = NA_integer_, n_docs = n_noise,
    purity = NA_real_, f1 = NA_real_,
    keywords = paste(top10_noise, collapse = ", "),
    stringsAsFactors = FALSE
  )
  for (ki in 1:10) noise_df[[paste0("kw", ki)]] <- top10_noise[ki]
  save_artifact(noise_df, file.path(PATHS$results, "noise_keywords"))
}

# =============================================================================
# Section 6: Compilation and Export of Keyword Analysis (Table 4)
# Formats the semantic profiles into a standardized publication-ready table.
# =============================================================================
cat("\n6. Assembling Table 4...\n")

disease_order <- c(
  "Neoplasms", "Digestive System Diseases", "Nervous System Diseases",
  "Cardiovascular Diseases", "General Pathological Conditions", "Unassigned"
)
table4_data <- keyword_df %>%
  arrange(match(assigned_disease, disease_order)) %>%
  dplyr::select(cluster, assigned_disease, n_docs, purity, f1, keywords)

if (!is.null(noise_df)) {
  table4_data <- bind_rows(
    table4_data,
    dplyr::select(noise_df, cluster, assigned_disease, n_docs, purity, f1, keywords)
  )
}

cat("\n   ── TABLE 4 ──\n")
print(table4_data, row.names = FALSE)

table4 <- table4_data %>%
  rename(Cluster  = cluster, Disease  = assigned_disease,
         N        = n_docs,  Purity   = purity,
         F1       = f1,      Keywords = keywords)

save_artifact(table4,          file.path(PATHS$results, "table4"))
save_artifact(full_keyword_df, file.path(PATHS$results, "cluster_keywords_full"))
save_artifact(keyword_df,      file.path(PATHS$results, "cluster_keywords_wide"))

# =============================================================================
# Section 7: Contextual Synthesis of Thematic Integrity
# Computes aggregate measures of clustering effectiveness compared against 
# supervised baselines to substantiate the findings in the paper.
# =============================================================================
cat(sprintf("\n── KEYWORDS SUMMARY (§4.4) ──\n"))
cat(sprintf("   Best experiment    : %s (ARI=%.4f)\n", best_exp_name, best_row$ARI))
cat(sprintf("   Clusters profiled  : %d\n", nrow(keyword_df)))
cat(sprintf("   Assigned clusters  : %d / %d (mapped to disease class)\n",
            sum(!is.na(keyword_df$assigned_class)), nrow(keyword_df)))
cat(sprintf("   Mean cluster purity: %.4f\n", mean(keyword_df$purity, na.rm = TRUE)))

valid_f1 <- keyword_df$f1[!is.na(keyword_df$f1)]
cat(sprintf("   Mean cluster F1    : %.4f\n", if (length(valid_f1) > 0) mean(valid_f1) else NaN))

macro_f1_best <- mean(valid_f1, na.rm = TRUE)
supervised_f1 <- 0.573
cat(sprintf("\n   Supervised baseline (Schopf et al. 2022): %.1f%%\n", supervised_f1 * 100))
cat(sprintf("   Our best Macro-F1 (5 assigned clusters)  : %.1f%%\n", macro_f1_best * 100))
cat(sprintf("   Gap (unsupervised vs supervised)         : %.1f pp\n",
            (supervised_f1 - macro_f1_best) * 100))

if (!is.null(noise_df)) {
  cat(sprintf("   Noise docs         : %d (%.1f%% of corpus)\n",
              n_noise, 100 * n_noise / 7470))
}

cat(sprintf("\n── 06_keywords.R complete (%.1f sec)\n", elapsed(t0)))
cat("   results/table4.rds/.csv\n")
cat("   results/cluster_keywords_full.rds/.csv\n")
cat("   results/cluster_keywords_wide.rds/.csv\n")
if (!is.null(noise_df)) cat("   results/noise_keywords.rds/.csv\n")
cat("   Next: source('R/07_figures.R')\n")