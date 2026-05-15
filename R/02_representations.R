# =============================================================================
# 02_representations.R — Text Representations: TF-IDF + PubMedBERT
# BioCluster-R | ICCA 2026
#
# Inputs  : data/processed/corpus_balanced.rds
#           data/embeddings/embeddings_pubmedbert.parquet
#
# Outputs : data/repr/repr_tfidf.rds       ← sparse 7470 × V (L2 normalised)
#           data/repr/vocab_tfidf.rds       ← pruned vocabulary (for c-TF-IDF §3.9)
#           data/repr/repr_pubmedbert.rds   ← dense 7470 × 768 (L2 normalised)
#
# Next    : 03_umap_pca.R
# =============================================================================

source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(text2vec)
  library(Matrix)
  library(arrow)
})

set.seed(42)
t0 <- proc.time()
cat("── 02_representations.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# =============================================================================
# Section 0: Corpus Initialization and Validation
# Loads the balanced and preprocessed corpus, verifying row counts and document 
# order to ensure precise alignment before generating text representations.
# =============================================================================
cat("0. Loading corpus_balanced.rds...\n")

corpus <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))

stopifnot(
  "Missing columns" = all(c("doc_id", "label", "text_clean", "text_raw") %in% names(corpus)),
  "Row count wrong" = nrow(corpus) == 7470,
  "doc_id[1] wrong" = corpus$doc_id[1] == "doc1",
  "doc_id[N] wrong" = tail(corpus$doc_id, 1) == paste0("doc", nrow(corpus))
)

corpus_ids <- corpus$doc_id   # authoritative order: doc1 … doc7470
cat(sprintf("   %d documents | doc1 → doc%d ✓\n\n", nrow(corpus), nrow(corpus)))

# =============================================================================
# Section 1: TF-IDF Representation (Track A)
# Constructs a Bag-of-Words representation using lemmatized text, followed by 
# vocabulary pruning and Term Frequency-Inverse Document Frequency weighting.
# The resulting document-term matrix is L2-normalized for distance metrics.
# =============================================================================
cat("1. Building TF-IDF representation...\n")
t1 <- proc.time()

# Initialize tokenizer function for corpus parsing
make_itoken <- function(texts, ids) {
  itoken(texts, tokenizer = word_tokenizer, ids = ids, progressbar = FALSE)
}

# Vocabulary from unigrams
vocab_raw    <- create_vocabulary(make_itoken(corpus$text_clean, corpus_ids))
vocab_pruned <- prune_vocabulary(
  vocab_raw,
  term_count_min     = 5L,
  doc_proportion_max = 0.50
)
cat(sprintf("   Raw vocab: %d | Pruned vocab: %d terms\n",
            nrow(vocab_raw), nrow(vocab_pruned)))

if (!dplyr::between(nrow(vocab_pruned), 5000L, 60000L)) {
  warning("Vocabulary size outside expected range 5K–60K: ", nrow(vocab_pruned))
}

# Vectoriser from pruned vocabulary
vectorizer <- vocab_vectorizer(vocab_pruned)

# Document-Term Matrix (raw counts, sparse)
dtm <- create_dtm(make_itoken(corpus$text_clean, corpus_ids), vectorizer)
stopifnot("DTM row count wrong" = nrow(dtm) == 7470)
cat(sprintf("   DTM: %d × %d (sparse)\n", nrow(dtm), ncol(dtm)))

# TF-IDF weighting: TF × log(N / df_t)
tfidf_model <- TfIdf$new()
dtm_tfidf   <- fit_transform(dtm, tfidf_model)

# Apply L2 normalization to create unit-length document vectors
# Ensures cosine similarity aligns with Euclidean distance (required for UMAP)
row_norms               <- sqrt(Matrix::rowSums(dtm_tfidf^2))
row_norms[row_norms == 0] <- 1   # guard: empty doc after filtering keeps zero vector
repr_tfidf              <- dtm_tfidf / row_norms
rownames(repr_tfidf)    <- corpus_ids

# Verify norms
norm_check <- sqrt(Matrix::rowSums(repr_tfidf^2))
cat(sprintf("   L2 norm check — min: %.6f | mean: %.6f | max: %.6f\n",
            min(norm_check), mean(norm_check), max(norm_check)))

saveRDS(repr_tfidf,   file.path(PATHS$repr, "repr_tfidf.rds"))
saveRDS(vocab_pruned, file.path(PATHS$repr, "vocab_tfidf.rds"))

cat(sprintf("   Saved: repr_tfidf.rds (%d × %d) | vocab_tfidf.rds (%d terms)\n",
            nrow(repr_tfidf), ncol(repr_tfidf), nrow(vocab_pruned)))
cat(sprintf("   TF-IDF elapsed: %.1f sec\n\n", elapsed(t1)))

# =============================================================================
# Section 2: PubMedBERT Representation (Track B)
# Loads high-dimensional contextual embeddings pre-computed via PubMedBERT.
# Performs mandatory data alignment and vector norm checks before final export.
# =============================================================================
cat("2. Loading PubMedBERT parquet...\n")
t2 <- proc.time()

parquet_path <- file.path(PATHS$embeddings, "embeddings_pubmedbert.parquet")
if (!file.exists(parquet_path)) {
  stop(
    "Parquet file not found: ", parquet_path, "\n",
    "Run: python python/embed_pubmedbert.py"
  )
}

pubmed_df <- arrow::read_parquet(parquet_path)
cat(sprintf("   Parquet loaded: %d rows × %d cols\n",
            nrow(pubmed_df), ncol(pubmed_df)))

# Extract embedding dimensions (dim_0 … dim_767)
dim_cols <- grep("^dim_", names(pubmed_df), value = TRUE)
stopifnot(
  "Wrong row count in parquet"     = nrow(pubmed_df) == 7470,
  "Wrong embedding dimensions"     = length(dim_cols) == 768
)

repr_pubmedbert          <- as.matrix(pubmed_df[, dim_cols])
rownames(repr_pubmedbert) <- pubmed_df$doc_id

cat(sprintf("   Matrix: %d × %d\n", nrow(repr_pubmedbert), ncol(repr_pubmedbert)))

# --- Alignment Verification --------------------------------------------------
# Ensures document ID order in parquet precisely matches the processed corpus
if (!all(pubmed_df$doc_id == corpus_ids)) {
  mismatch_idx <- which(pubmed_df$doc_id != corpus_ids)
  stop(
    "doc_id MISMATCH between parquet and corpus_balanced.\n",
    "First mismatch at row: ", mismatch_idx[1], "\n",
    "Parquet doc_id: ",   pubmed_df$doc_id[mismatch_idx[1]], "\n",
    "Corpus doc_id:  ", corpus_ids[mismatch_idx[1]]
  )
}
cat("   doc_id alignment: all 7470 rows match corpus order ✓\n")

# --- L2 Norm Verification ----------------------------------------------------
# Confirms generated embeddings are correctly L2-normalized
pubmed_norms <- sqrt(rowSums(repr_pubmedbert^2))
norm_ok      <- all(abs(pubmed_norms - 1.0) < 0.001)
cat(sprintf("   L2 norm check — min: %.6f | max: %.6f | within 0.001: %s\n",
            min(pubmed_norms), max(pubmed_norms),
            if (norm_ok) "✓" else "FAIL"))
if (!norm_ok) stop("PubMedBERT L2 norm check failed — re-generate embeddings")

saveRDS(repr_pubmedbert, file.path(PATHS$repr, "repr_pubmedbert.rds"))
cat(sprintf("   Saved: repr_pubmedbert.rds (%d × %d)\n",
            nrow(repr_pubmedbert), ncol(repr_pubmedbert)))
cat(sprintf("   PubMedBERT load elapsed: %.1f sec\n\n", elapsed(t2)))

# =============================================================================
# Section 3: Data Export and Summary Logging
# Finalizes generation and logs representation matrix dimensions for validation.
# =============================================================================
cat("3. Final verification...\n")

repr_list <- list(tfidf = repr_tfidf, pubmedbert = repr_pubmedbert)

for (name in names(repr_list)) {
  m  <- repr_list[[name]]
  rn <- rownames(m)
  cat(sprintf("   %-12s : %d × %-5d | doc_id aligned: %s\n",
              name,
              nrow(m), ncol(m),
              if (all(rn == corpus_ids)) "✓" else "FAIL"))
}

cat(sprintf("\n── 02_representations.R complete (%.1f sec)\n", elapsed(t0)))
cat("   Saved to data/repr/:\n")
cat("     repr_tfidf.rds, vocab_tfidf.rds, repr_pubmedbert.rds\n")
cat("   Next: source('R/03_umap_pca.R')\n")
