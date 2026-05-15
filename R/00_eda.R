# =============================================================================
# 00_eda.R — Exploratory Data Analysis & Biomedical Stopword Derivation
# BioCluster-R | ICCA 2026
#
# Purpose:
#   This is the initial stage of the pipeline. It merges the raw data, evaluates
#   the class distributions to determine subsampling sizes, calculates document
#   lengths, and performs a sophisticated Coefficient of Variation (CV) analysis
#   to derive dataset-specific, non-discriminative biomedical stopwords.
#
# Inputs  : data/raw/medical_tc_train.csv
#           data/raw/medical_tc_test.csv
#
# Outputs : data/processed/corpus_full.rds/.csv
#           data/processed/eda_class_dist.rds/.csv
#           data/processed/eda_length_stats.rds/.csv
#           data/processed/eda_cv_analysis.rds/.csv
#           data/processed/biomedical_stopwords.rds/.csv  ← key output
#           figures/eda_s1_class_dist.pdf/.png
#           figures/eda_s2_cv_analysis.pdf/.png
# =============================================================================

# -----------------------------------------------------------------------------
# INITIALIZATION & LIBRARY IMPORTS
# Sources the shared project paths from utils.R and loads necessary libraries
# strictly suppressing startup messages to keep execution clean.
# -----------------------------------------------------------------------------
source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(quanteda)
  library(stringi)
})

# -----------------------------------------------------------------------------
# RANDOM SEED AND DEVICE SETUP
# The random seed is fixed for deterministic reproducibility. A timing capture
# is initiated. The PDF device is selected intelligently based on system
# capabilities to ensure high-quality vector graphics rendering.
# -----------------------------------------------------------------------------
set.seed(42)
t0 <- proc.time()
cat("── 00_eda.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

pdf_device <- if (isTRUE(capabilities()["cairo"])) cairo_pdf else pdf
if (isTRUE(capabilities()["cairo"])) {
  cat("   Device: cairo_pdf enabled for high-quality vector rendering.\n\n")
} else {
  cat("   Device: Standard pdf fallback (cairo unavailable).\n\n")
}

# =============================================================================
# 1. LOAD AND MERGE RAW DATA
#
# Purpose:
#   Reads the raw, unprocessed training and testing sets from the Medical
#   Abstracts TC Corpus. Since clustering is an unsupervised machine learning
#   task, the original supervised train/test splits are irrelevant. This step
#   merges them into a single comprehensive dataset.
#
# Logic & Parameters:
#   - col_types is explicit to prevent parsing issues on character/integer data.
#   - filter ensures empty or NA abstracts are stripped from the pipeline early.
#   - doc_id is synthesized using row_number() for unique future referencing.
#   - word_count is derived quickly via regex counting non-whitespace blocks.
# =============================================================================
cat("1. Loading raw CSV datasets...\n")
train_raw <- read_csv(
  file.path(PATHS$raw, "medical_tc_train.csv"),
  col_types = cols(condition_label = col_integer(), medical_abstract = col_character()),
  show_col_types = FALSE
)
test_raw <- read_csv(
  file.path(PATHS$raw, "medical_tc_test.csv"),
  col_types = cols(condition_label = col_integer(), medical_abstract = col_character()),
  show_col_types = FALSE
)
cat(sprintf("   Train dataset: %d rows loaded.\n", nrow(train_raw)))
cat(sprintf("   Test dataset: %d rows loaded.\n", nrow(test_raw)))

corpus_full <- bind_rows(train_raw, test_raw) %>%
  rename(label = condition_label, text_raw = medical_abstract) %>%
  filter(!is.na(text_raw), text_raw != "") %>%
  mutate(
    disease_name = DISEASE_MAP[as.character(label)],
    doc_id       = paste0("doc", row_number()),
    word_count   = str_count(text_raw, "\\S+")
  ) %>%
  select(doc_id, label, disease_name, text_raw, word_count)

cat(sprintf(
  "   Merged corpus created: %s valid documents across %d classes.\n\n",
  format(nrow(corpus_full), big.mark = ","), n_distinct(corpus_full$label)
))

save_artifact(corpus_full, file.path(PATHS$processed, "corpus_full"))

# =============================================================================
# 2. CLASS DISTRIBUTION ANALYSIS
#
# Purpose:
#   To compute the frequency of documents across the five disease classes.
#   This identifies the minority class which acts as the maximum bounds for
#   the balanced subsampling procedure required later.
#
# Logic & Outputs:
#   - group/count on 'label' generates 'n' and derives percentages ('pct').
#   - min_n isolates the smallest cluster size.
#   - balanced_n determines the total theoretical size of an perfectly evenly
#     distributed corpus.
# =============================================================================
cat("2. Analyzing class distribution...\n")

class_dist <- corpus_full %>%
  count(label, disease_name, name = "n") %>%
  mutate(pct = round(100 * n / sum(n), 2)) %>%
  arrange(label)

print(class_dist)

min_n <- min(class_dist$n)
min_class <- class_dist$disease_name[which.min(class_dist$n)]
balanced_n <- min_n * 5L

cat(sprintf("\n   Minority class identified: '%s' (%d documents)\n", min_class, min_n))
cat(sprintf(
  "   Downstream balanced corpus target: %s documents (%d per class × 5)\n\n",
  format(balanced_n, big.mark = ","), min_n
))

save_artifact(class_dist, file.path(PATHS$processed, "eda_class_dist"))

# =============================================================================
# 3. ABSTRACT LENGTH STATISTICS
#
# Purpose:
#   To calculate robust summary statistics concerning the verbosity of abstracts
#   grouped by disease category.
#
# Logic:
#   - Aggregates mean, median, min, max, and SD of 'word_count'.
#   - Counts the frequency of extremely short abstracts (under 20 words). These
#     are documented for methodological transparency but left intact per paper
#     design.
# =============================================================================
cat("3. Computing abstract length statistics...\n")

length_stats <- corpus_full %>%
  group_by(label, disease_name) %>%
  summarise(
    n = n(),
    mean_words = round(mean(word_count), 1),
    median_words = median(word_count),
    min_words = min(word_count),
    max_words = max(word_count),
    sd_words = round(sd(word_count), 1),
    n_short_lt20 = sum(word_count < 20),
    .groups = "drop"
  )

print(length_stats)
cat(sprintf(
  "\n   Extremely short abstracts (<20 words): %d found. (Documented, keeping intact)\n\n",
  sum(length_stats$n_short_lt20)
))

save_artifact(length_stats, file.path(PATHS$processed, "eda_length_stats"))

# =============================================================================
# 4. BIOMEDICAL STOPWORD DERIVATION — CV ANALYSIS
#
# Purpose:
#   Traditional NLP relies on standard stopword lists (e.g. 'the', 'and').
#   However, in specific domains like medicine, terms like 'patient' or 'study'
#   appear ubiquitously across all topics. If left in, they dilute TF-IDF
#   weights. This section uses a Coefficient of Variation (CV) metric to identify
#   and extract these highly frequent, non-discriminative terms.
#
# Process:
#   1. Minimal Tokenization: Strings are normalized (NFKC, lowercase, squish)
#      and standard English stopwords are removed.
#   2. Term Frequencies: Identifies the top 300 most common unigrams in the
#      entire corpus.
#   3. Relative Normalization: Given class imbalance, raw frequency across classes
#      is skewed. Frequencies are normalized per class by the total volume of
#      tokens in that class.
#   4. Coefficient of Variation: Computed as SD / Mean across the 5 classes for
#      each term. A low CV (< 0.25) implies the term is spread uniformly across
#      all categories and thus holds no clustering value.
# =============================================================================
cat("4. Deriving biomedical stopwords (CV analysis on top-300 terms)...\n")

CV_THRESHOLD <- 0.25
TOP_N <- 300

english_sw <- quanteda::stopwords("en", source = "snowball")
cat(sprintf("   Standard English stopword dictionary loaded (%d words).\n", length(english_sw)))

cat("   Tokenizing full corpus (lowercase, NFKC, alpha-only)...\n")
tokens_df <- corpus_full %>%
  transmute(
    label,
    token = str_to_lower(text_raw) %>%
      stri_trans_nfkc() %>%
      str_replace_all("[^a-z\\s]", " ") %>%
      str_squish() %>%
      str_split("\\s+")
  ) %>%
  unnest(token) %>%
  filter(nchar(token) > 1, !token %in% english_sw)

cat(sprintf("   Total valid tokens parsed: %s\n", format(nrow(tokens_df), big.mark = ",")))

class_token_totals <- tokens_df %>%
  count(label, name = "class_total_tokens")

cat(sprintf("   Isolating top %d most frequent global unigrams...\n", TOP_N))
top300 <- tokens_df %>%
  count(token, name = "global_freq") %>%
  slice_max(global_freq, n = TOP_N, with_ties = FALSE)

cat(sprintf("   Top 10 most frequent terms overall: %s\n", paste(head(top300$token, 10), collapse = ", ")))

cat("   Calculating per-class relative frequency & Coefficient of Variation (CV)...\n")
per_class_raw <- tokens_df %>%
  filter(token %in% top300$token) %>%
  count(label, token, name = "class_freq") %>%
  complete(label = 1:5, token = top300$token, fill = list(class_freq = 0))

per_class <- per_class_raw %>%
  left_join(class_token_totals, by = "label") %>%
  mutate(rel_freq = class_freq / class_total_tokens)

cv_analysis <- per_class %>%
  group_by(token) %>%
  summarise(
    cv_mean = mean(rel_freq),
    cv_sd   = sd(rel_freq),
    cv      = round(cv_sd / cv_mean, 4),
    .groups = "drop"
  ) %>%
  left_join(top300, by = "token") %>%
  left_join(
    per_class %>%
      mutate(col = paste0("relfreq_class_", label)) %>%
      pivot_wider(
        id_cols = token, names_from = col,
        values_from = rel_freq, values_fill = 0
      ),
    by = "token"
  ) %>%
  arrange(cv)

n_candidates <- sum(cv_analysis$cv < CV_THRESHOLD, na.rm = TRUE)
cat(sprintf("   Analysis complete: %d terms exhibit CV < %.2f.\n\n", n_candidates, CV_THRESHOLD))

save_artifact(cv_analysis, file.path(PATHS$processed, "eda_cv_analysis"))

# =============================================================================
# 5. FINAL BIOMEDICAL STOPWORD LIST
#
# Purpose:
#   Filters the CV analysis frame to extract the final biomedical stopword
#   dictionary based on the established threshold.
#
# Variables:
#   n_removed: The absolute number of tokens that this list will strip downstream.
#   pct_removed: The percentage impact on the token pool.
# =============================================================================
cat("5. Finalizing biomedical stopword list...\n")

biomedical_stopwords <- cv_analysis %>%
  filter(cv < CV_THRESHOLD) %>%
  select(term = token, cv, global_freq, starts_with("relfreq_class_")) %>%
  arrange(cv)

cat(sprintf("   Generated %d final biomedical stopwords:\n", nrow(biomedical_stopwords)))
cat(paste("  ", biomedical_stopwords$term, collapse = ", "), "\n")

n_removed <- sum(tokens_df$token %in% biomedical_stopwords$term)
pct_removed <- round(100 * n_removed / nrow(tokens_df), 2)
cat(sprintf(
  "\n   Impact projection: This will remove %s tokens (%.2f%% of the active token pool).\n\n",
  format(n_removed, big.mark = ","), pct_removed
))

save_artifact(biomedical_stopwords, file.path(PATHS$processed, "biomedical_stopwords"))

# =============================================================================
# 6. FIGURES & VISUALIZATIONS
#
# Purpose:
#   Generates auxiliary plots for the supplementary section of the paper,
#   specifically targeting the class distribution and the Coefficient of
#   Variation analysis distribution.
#
# Function:
#   save_fig: A localized scoped helper that uses ggsave to write both PDF and
#             high-res PNG graphics using the predefined cairo_pdf device.
# =============================================================================
cat("6. Rendering exploratory figures...\n")

save_fig <- function(p, name, w = 6.5, h = 4) {
  ggsave(file.path(PATHS$figures, paste0(name, ".pdf")),
    p,
    width = w, height = h, device = pdf_device
  )
  ggsave(file.path(PATHS$figures, paste0(name, ".png")),
    p,
    width = w, height = h, dpi = 300
  )
  cat(sprintf("   ✓ Saved figure: %s (%.1f x %.1f in)\n", name, w, h))
}

# -----------------------------------------------------------------------------
# FIGURE S1: CLASS DISTRIBUTION
# Horizontal bar chart mapped to the 5 Medical Abstracts TC categories.
# Subtitles explicitly map out total volume and minority constraints.
# -----------------------------------------------------------------------------
p_class <- class_dist %>%
  mutate(
    disease_short = str_replace(disease_name, " Diseases", " Dis.") %>%
      str_replace("General Pathological Conditions", "General Pathological"),
    label_txt = paste0(n, " (", pct, "%)")
  ) %>%
  ggplot(aes(x = reorder(disease_short, n), y = n)) +
  geom_col(fill = "#4472C4", width = 0.6) +
  geom_text(aes(label = label_txt), hjust = -0.08, size = 3, colour = "grey30") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title = "Class distribution — full corpus",
    subtitle = paste0(
      "14,438 abstracts | 5 MeSH disease categories\n",
      "Minority class: '", min_class, "' (", min_n,
      " docs) → balanced corpus = ", balanced_n
    ),
    x = NULL, y = "Number of abstracts"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9, colour = "grey40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 10)
  )

save_fig(p_class, "eda_s1_class_dist", w = 7, h = 4)

# -----------------------------------------------------------------------------
# FIGURE S2: CV ANALYSIS
# Visualizes the top 300 most frequent tokens across the corpus sorted ascending
# by their Coefficient of Variation. A horizontal boundary intercepts the plot
# at CV = 0.25 to show the strict cutoff threshold delineating stopwords.
# -----------------------------------------------------------------------------
p_cv <- cv_analysis %>%
  mutate(is_sw = cv < CV_THRESHOLD) %>%
  ggplot(aes(x = reorder(token, cv), y = cv, fill = is_sw)) +
  geom_col(width = 0.85, show.legend = TRUE) +
  geom_hline(
    yintercept = CV_THRESHOLD, linetype = "dashed",
    colour = "#C0392B", linewidth = 0.7
  ) +
  annotate("text",
    x = n_candidates / 2,
    y = CV_THRESHOLD + 0.02,
    label = paste0("CV threshold = ", CV_THRESHOLD),
    colour = "#C0392B", size = 2.8, hjust = 0
  ) +
  scale_fill_manual(
    values = c("FALSE" = "#4472C4", "TRUE" = "#E05C4B"),
    labels = c(
      paste0("Kept (n = ", TOP_N - n_candidates, ")"),
      paste0(
        "Biomedical stopword, CV < ", CV_THRESHOLD,
        " (n = ", n_candidates, ")"
      )
    )
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(
    title = paste0("CV analysis — top-", TOP_N, " unigrams"),
    subtitle = paste0(
      "Relative frequency (normalized by class token count) across 5 disease classes\n",
      "Removal impact: ", pct_removed, "% of post-English-stopword tokens"
    ),
    x = paste0("Terms sorted by CV ascending (", TOP_N, " terms, labels hidden)"),
    y = "CV = SD / mean of per-class relative frequency",
    fill = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 8, colour = "grey40"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.4, "cm")
  )

save_fig(p_cv, "eda_s2_cv_analysis", w = 7, h = 4.5)

# =============================================================================
# Summary Execution Trace
# =============================================================================
cat(sprintf("\n── 00_eda.R successfully executed (%.1f sec) ──\n", elapsed(t0)))
cat(sprintf("   Pipeline Constraints Generated:\n"))
cat(sprintf("   - Subsample Cap : %s documents per class\n", format(min_n, big.mark = ",")))
cat(sprintf("   - Total Target  : %s balanced documents\n", format(balanced_n, big.mark = ",")))
cat(sprintf("   - Stopwords     : %d terms derived programmatically\n", nrow(biomedical_stopwords)))
cat("\n   Next recommended execution: source('R/01_preprocessing.R')\n")
# End of Script
