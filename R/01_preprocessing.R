# =============================================================================
# 01_preprocessing.R — 7-Step Text Preprocessing Pipeline
# BioCluster-R | ICCA 2026
#
# Inputs  : data/processed/corpus_full.rds
#           data/processed/biomedical_stopwords.rds
#           english-ewt-ud-2.5-191206.udpipe  (searched in common locations)
#
# Outputs : data/processed/corpus_balanced.rds     ← Track A + Track B
#           data/processed/track_b_for_python.csv  ← raw text for Python embedding
#           data/processed/preprocessing_stats.rds/.csv
#
# Run time: ~15 minutes (udpipe lemmatisation dominates)
# Next    : python/embed_pubmedbert.py
# =============================================================================

source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(udpipe)
  library(quanteda)
  library(stringi)
})

set.seed(42)
t0 <- proc.time()
cat("── 01_preprocessing.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# =============================================================================
# Section 1: Data Initialization and Resource Loading
# Initializes the dataset and required language models for the preprocessing 
# pipeline, including standard English and biomedical-specific stopwords.
# =============================================================================
cat("1. Loading inputs...\n")

corpus_full <- readRDS(file.path(PATHS$processed, "corpus_full.rds"))
bio_sw      <- readRDS(file.path(PATHS$processed, "biomedical_stopwords.rds"))
english_sw  <- quanteda::stopwords("en", source = "snowball")
all_stopwords <- union(english_sw, bio_sw$term)

# Locate udpipe model — checks project root and data/raw/ (both are common)
udmodel_file <- "english-ewt-ud-2.5-191206.udpipe"
udmodel_candidates <- c(
  udmodel_file,                                # project root (default download location)
  file.path(PATHS$raw, udmodel_file),          # data/raw/
  file.path("data", udmodel_file)              # data/
)
udmodel_path <- Find(file.exists, udmodel_candidates)

if (is.null(udmodel_path)) {
  stop(
    "udpipe model not found. Download it once by running:\n",
    "  library(udpipe)\n",
    "  udpipe_download_model('english-ewt')   # saves to project root\n"
  )
}
cat(sprintf("   udpipe model: %s\n", udmodel_path))

cat(sprintf("   corpus_full     : %d docs\n",   nrow(corpus_full)))
cat(sprintf("   english_sw      : %d terms\n",  length(english_sw)))
cat(sprintf("   biomedical_sw   : %d terms\n",  length(bio_sw$term)))
cat(sprintf("   combined_sw     : %d terms\n\n", length(all_stopwords)))

# =============================================================================
# Section 2: Stratified Subsampling for Corpus Balancing
# Generates a class-balanced corpus based on the minority class count to prevent 
# model bias during representation learning and clustering (Methodology Section 3.1).
# =============================================================================
cat("2. Stratified subsampling...\n")

n_per_class <- min(count(corpus_full, label)$n)

set.seed(42)
corpus_balanced <- corpus_full %>%
  group_by(label) %>%
  slice_sample(n = n_per_class, replace = FALSE) %>%
  ungroup() %>%
  arrange(label) %>%
  mutate(doc_id = paste0("doc", row_number()))   # doc1 → docN

stopifnot(
  "Class balance check failed" = all(table(corpus_balanced$label) == n_per_class)
)

cat(sprintf("   n per class : %d\n", n_per_class))
cat(sprintf("   total docs  : %d\n", nrow(corpus_balanced)))
cat(sprintf("   doc_id range: %s → %s\n\n",
            corpus_balanced$doc_id[1], tail(corpus_balanced$doc_id, 1)))

# =============================================================================
# Section 3: Initial Vectorized Preprocessing (Steps 1-3)
# Performs lowercase conversion, Unicode NFKC normalization, and non-alphabetic 
# character removal. Track B remains unmodified for PubMedBERT's internal tokenization.
# =============================================================================
cat("3. Preprocessing steps 1–3 (vectorised)...\n")

corpus_balanced <- corpus_balanced %>%
  mutate(
    # Step 1: Lowercase
    text_s1 = str_to_lower(text_raw),

    # Step 2: Unicode NFKC normalisation
    # Resolves accented chars, special dashes, non-breaking spaces,
    # chemical notation common in PubMed abstracts
    text_s2 = stri_trans_nfkc(text_s1),

    # Step 3: Remove non-alphabetic characters (numbers, punctuation, symbols)
    # These carry no disease-category discriminative signal
    text_s3 = str_squish(str_replace_all(text_s2, "[^a-z\\s]", " "))
  )

# Save intermediate checkpoint before computational-heavy lemmatization step
saveRDS(corpus_balanced, file.path(PATHS$processed, "corpus_prelemma.rds"))
cat("   Steps 1–3 complete | Checkpoint: corpus_prelemma.rds\n\n")

# =============================================================================
# Section 4: Lemmatization and Part-of-Speech Filtering
# Applies udpipe for semantic lemmatization. Restricts to informative POS tags 
# (NOUN, VERB, ADJ) and filters tokens by length and stopword criteria.
# =============================================================================
cat("4. Lemmatisation with udpipe (~15 min)...\n")
t_udpipe <- proc.time()

udmodel <- udpipe_load_model(udmodel_path)

lemmatise_batched <- function(texts, doc_ids, model, batch_size = 500L) {
  n_batches <- ceiling(length(texts) / batch_size)
  results   <- vector("list", n_batches)

  for (i in seq_len(n_batches)) {
    idx_start <- (i - 1L) * batch_size + 1L
    idx_end   <- min(i * batch_size, length(texts))

    if (i %% 5L == 1L || i == n_batches) {
      cat(sprintf("   Batch %d/%d (docs %d–%d)...\n",
                  i, n_batches, idx_start, idx_end))
    }

    ann <- udpipe_annotate(
      model,
      x      = texts[idx_start:idx_end],
      doc_id = doc_ids[idx_start:idx_end],
      parser = "none"
    )
    results[[i]] <- as.data.frame(ann, detailed = FALSE)
  }
  bind_rows(results)
}

annotation <- lemmatise_batched(
  texts      = corpus_balanced$text_s3,
  doc_ids    = corpus_balanced$doc_id,
  batch_size = 500L,
  model      = udmodel
)

cat(sprintf("\n   udpipe complete: %.1f sec | raw token rows: %s\n",
            elapsed(t_udpipe), format(nrow(annotation), big.mark = ",")))

# =============================================================================
# Section 5: Token Verification and Document Reconstruction (Track A)
# Verifies stopword constraints and reconstructs the clean textual corpus 
# for subsequent traditional Bag-of-Words and TF-IDF representations.
# =============================================================================
cat("5. Filtering tokens and rebuilding clean documents...\n")

clean_tokens <- annotation %>%
  filter(
    upos %in% c("NOUN", "VERB", "ADJ"),
    nchar(lemma) > 2L,
    str_detect(lemma, "^[a-z]+$"),
    !tolower(token) %in% all_stopwords,
    !tolower(lemma) %in% all_stopwords
  ) %>%
  mutate(lemma_clean = tolower(lemma))

tokens_kept  <- nrow(clean_tokens)
tokens_total <- nrow(annotation)
pct_kept     <- round(100 * tokens_kept / tokens_total, 1)

cat(sprintf("   Kept: %s / %s tokens (%.1f%%)\n",
            format(tokens_kept, big.mark = ","),
            format(tokens_total, big.mark = ","),
            pct_kept))

# Reconstruct one clean string per document (Track A)
track_a <- clean_tokens %>%
  group_by(doc_id) %>%
  summarise(
    text_clean       = paste(lemma_clean, collapse = " "),
    word_count_clean = n(),
    .groups = "drop"
  )

corpus_balanced <- corpus_balanced %>%
  left_join(track_a, by = "doc_id")

# =============================================================================
# Section 6: Quality Assurance and Data Validation
# Ensures logical integrity, verifies document order, detects empty documents, 
# and computes summary statistics for the processed vocabulary.
# =============================================================================
cat("\n6. Verification...\n")

# Verify document ID order alignment for subsequent Python Parquet integration
expected_ids <- paste0("doc", seq_len(nrow(corpus_balanced)))
stopifnot("doc_id ordering wrong" = all(corpus_balanced$doc_id == expected_ids))
cat(sprintf("   doc_id order   : %s → %s ✓\n",
            corpus_balanced$doc_id[1], tail(corpus_balanced$doc_id, 1)))

# Near-empty documents after preprocessing
empty_docs <- corpus_balanced %>% filter(is.na(text_clean) | word_count_clean < 5)
cat(sprintf("   Near-empty docs (<5 tokens): %d%s\n",
            nrow(empty_docs),
            if (nrow(empty_docs) > 0) " ← WARNING" else " ✓"))
if (nrow(empty_docs) > 0) {
  print(select(empty_docs, doc_id, label, disease_name, word_count_clean))
}

# Vocabulary size
all_lemmas <- unlist(str_split(corpus_balanced$text_clean, " "))
vocab_size  <- length(unique(all_lemmas[all_lemmas != ""]))
in_range    <- dplyr::between(vocab_size, 5000L, 80000L)
cat(sprintf("   Vocabulary size: %s unique lemmas %s\n",
            format(vocab_size, big.mark = ","),
            if (in_range) "✓" else "← WARNING: outside expected range 5K–80K"))

# Post-preprocessing word count by class
postpre_stats <- corpus_balanced %>%
  group_by(disease_name) %>%
  summarise(
    mean_clean   = round(mean(word_count_clean,   na.rm = TRUE), 1),
    median_clean = median(word_count_clean, na.rm = TRUE),
    min_clean    = min(word_count_clean,    na.rm = TRUE),
    max_clean    = max(word_count_clean,    na.rm = TRUE),
    .groups = "drop"
  )
cat("\n   Post-preprocessing word count by class:\n")
print(postpre_stats)

# =============================================================================
# Section 7: Final Data Export and Checkpointing
# Exports the balanced text for Track A and raw text for Track B (Python).
# Also saves comprehensive preprocessing statistics for analysis.
# =============================================================================
cat("\n7. Saving outputs...\n")

corpus_final <- corpus_balanced %>%
  select(
    doc_id, label, disease_name,
    text_raw, text_clean,
    word_count_raw = word_count,
    word_count_clean
  ) %>%
  arrange(as.integer(str_remove(doc_id, "doc")))

saveRDS(corpus_final, file.path(PATHS$processed, "corpus_balanced.rds"))
cat(sprintf("   corpus_balanced.rds : %d rows × %d cols\n",
            nrow(corpus_final), ncol(corpus_final)))

# Track B CSV — for Python embedding script (doc_id, label, text_raw only)
write_csv(
  select(corpus_final, doc_id, label, text_raw),
  file.path(PATHS$processed, "track_b_for_python.csv")
)
cat(sprintf("   track_b_for_python.csv : %d rows\n", nrow(corpus_final)))

# Preprocessing summary for reference and paper §4.1
preprocessing_stats <- tibble(
  metric = c(
    "full_corpus_docs", "n_per_class", "balanced_total_docs",
    "n_english_sw", "n_biomedical_sw", "n_combined_sw",
    "udpipe_raw_tokens", "tokens_kept", "pct_tokens_kept",
    "vocab_size_unique_lemmas", "n_empty_docs_lt5_tokens"
  ),
  value = c(
    nrow(corpus_full), n_per_class, nrow(corpus_final),
    length(english_sw), length(bio_sw$term), length(all_stopwords),
    tokens_total, tokens_kept, pct_kept,
    vocab_size, nrow(empty_docs)
  )
)
save_artifact(preprocessing_stats,
              file.path(PATHS$processed, "preprocessing_stats"))
cat("   preprocessing_stats.rds/.csv saved\n")

# =============================================================================
# Section 8: Post-Export Data Integrity Verification
# Re-loads the exported artifacts to ensure successful disk operations.
# =============================================================================
cat("\n── Final verification (re-reading from disk)...\n")
chk <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))

stopifnot(
  "Row count wrong"    = nrow(chk) == nrow(corpus_final),
  "Class balance wrong"= all(table(chk$label) == n_per_class),
  "doc_id start wrong" = chk$doc_id[1] == "doc1",
  "doc_id end wrong"   = tail(chk$doc_id, 1) == paste0("doc", nrow(chk)),
  "Track A NA found"   = sum(is.na(chk$text_clean)) == 0,
  "Track B NA found"   = sum(is.na(chk$text_raw))   == 0
)

cat(sprintf("   %d docs | %d per class | doc1 → doc%d | Track A + B clean ✓\n",
            nrow(chk), n_per_class, nrow(chk)))

cat(sprintf("\n── 01_preprocessing.R complete (%.1f sec)\n", elapsed(t0)))
cat("   Next: python python/embed_pubmedbert.py\n")
