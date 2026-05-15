# =============================================================================
# 07_figures.R  —  All Figures: Primary (2–4, S1–S2) + Additional (5–8)
# BioCluster-R  |  ICCA 2026
#
# PRIMARY
#   fig2_pca_vs_umap          2×2 scatter: PCA vs UMAP coloured by disease
#   fig3_ari_comparison       Grouped bar: ARI by representation × algorithm
#   fig4_hdbscan_cluster_map  UMAP 2D: TF-IDF HDBSCAN clusters (best exp.)
#   figs1_kmeans_sweep        Silhouette sweep k ∈ {2…10}
#   figs2_hdbscan_grid        minPts grid with k>20 exclusion zone
#
# ADDITIONAL
#   fig5_confusion_matrix     Cluster × disease contingency (row-normalised %)
#   fig6_ctfidf_heatmap       c-TF-IDF top-5 terms per cluster heatmap
#   fig7_ari_noise_bubble     ARI vs noise trade-off bubble chart
#   fig8_per_class_f1         Per-disease F1 across all 6 experiments
#
# Rules:
#   - No figure numbers inside plots (journals add them to captions)
#   - No text overflow / overlap
#   - Fig 2 and Fig 4 share the same parent-disease colour family
#   - Fig 2 subsampled: 300/class × 5 = 1,500 points  (readable density)
#   - All legends compact and inside figure dimensions
# =============================================================================

source("R/utils.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

set.seed(42)
t0 <- proc.time()
cat("── 07_figures.R started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

PDF_DEV <- if (isTRUE(capabilities()["cairo"])) cairo_pdf else pdf

save_fig <- function(p, name, w, h) {
  ggsave(file.path(PATHS$figures, paste0(name, ".pdf")),
    p,
    width = w, height = h, device = PDF_DEV
  )
  ggsave(file.path(PATHS$figures, paste0(name, ".png")),
    p,
    width = w, height = h, dpi = 300
  )
  cat(sprintf("   ✓ %-30s  %.1f × %.1f in\n", name, w, h))
}

# =============================================================================
# Section 0.1: Standardized Plotting Theme
# Defines a consistent, publication-ready minimal theme applied across all
# generated figures to ensure uniform typography, margins, and grid spacing.
# =============================================================================
theme_paper <- function(base = 9) {
  theme_minimal(base_size = base) +
    theme(
      plot.title = element_text(
        size = base + 0.5, face = "bold",
        margin = margin(b = 3)
      ),
      plot.subtitle = element_text(
        size = base - 1.5, colour = "grey45",
        margin = margin(b = 3)
      ),
      axis.title = element_text(size = base - 0.5),
      axis.text = element_text(size = base - 1.5, colour = "grey35"),
      legend.title = element_text(size = base - 0.5, face = "bold"),
      legend.text = element_text(size = base - 1.5),
      legend.key.size = unit(0.30, "cm"),
      legend.background = element_blank(),
      legend.margin = margin(t = 2),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey93", linewidth = 0.3),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(6, 6, 6, 6)
    )
}

# =============================================================================
# Section 0.2: Global Color Ontology
# Establishes a strictly aligned color palette between true disease classes
# (parent colors) and algorithmically derived clusters (shaded sub-variants).
# =============================================================================

# Five disease classes  →  used in Fig 2, Fig 5, Fig 8
DISEASE_COLORS <- c(
  "Cardiovascular"    = "#377EB8",
  "Digestive"         = "#FF7F00",
  "Gen. Path."        = "#4DAF4A",
  "Neoplasms"         = "#E41A1C",
  "Nervous"           = "#984EA3"
)

# Eight HDBSCAN clusters + Noise  →  used in Fig 4, Fig 5, Fig 6
# Colours derived from parent disease family where applicable
CLUSTER_COLORS <- c(
  "Noise"             = "grey55",
  "Neoplasms"         = "#E41A1C",
  "Digestive"         = "#FF7F00",
  "IBD"               = "#F1C40F", # Distinguishable yellow
  "Gen. Path."        = "#4DAF4A",
  "Cardiovascular"    = "#377EB8",
  "Nervous"           = "#984EA3",
  "Neurodegen."       = "#1ABC9C", # Distinguishable turquoise
  "Palliative"        = "#E84393" # Distinguishable pink
)

# Ordered factor levels for clusters in plots
CL_LEVELS <- names(CLUSTER_COLORS)[-1] # exclude Noise; add back where needed

# Integer cluster id → short name
CLUSTER_NAMES <- c(
  "1" = "Gen. Path.", "2" = "Digestive", "3" = "IBD",
  "4" = "Cardiovascular", "5" = "Nervous",
  "6" = "Neurodegen.", "7" = "Palliative", "8" = "Neoplasms"
)

# Integer class id → short disease name (for axes/labels)
DISEASE_SHORT <- c(
  "1" = "Neoplasms", "2" = "Digestive", "3" = "Nervous",
  "4" = "Cardiovascular", "5" = "Gen. Path."
)

REP_COL <- c("TF-IDF" = "#E05C4B", "PubMedBERT" = "#4472C4")

# =============================================================================
# Section 0.3: Data and Artifact Import
# Loads all necessary statistical summaries, embedding matrices, and vocabulary
# mappings required for comprehensive visual generation.
# =============================================================================
cat("0. Loading inputs...\n")

corpus <- readRDS(file.path(PATHS$processed, "corpus_balanced.rds"))
table3 <- readRDS(file.path(PATHS$results, "table3.rds"))
per_class_f1 <- readRDS(file.path(PATHS$results, "per_class_f1.rds"))
full_kw <- readRDS(file.path(PATHS$results, "cluster_keywords_full.rds"))

pca2 <- list(
  tfidf      = readRDS(file.path(PATHS$umap, "pca2_tfidf.rds")),
  pubmedbert = readRDS(file.path(PATHS$umap, "pca2_pubmedbert.rds"))
)
umap2 <- list(
  tfidf      = readRDS(file.path(PATHS$umap, "umap2_tfidf.rds")),
  pubmedbert = readRDS(file.path(PATHS$umap, "umap2_pubmedbert.rds"))
)
labels_hdb <- readRDS(file.path(PATHS$clustering, "labels_hdbscan_tfidf.rds"))

sweep_tfidf <- readRDS(file.path(PATHS$clustering, "kmeans_sweep_tfidf.rds"))
sweep_pbmed <- readRDS(file.path(PATHS$clustering, "kmeans_sweep_pubmedbert.rds"))
grid_tfidf <- readRDS(file.path(PATHS$clustering, "hdbscan_grid_tfidf.rds"))
grid_pbmed <- readRDS(file.path(PATHS$clustering, "hdbscan_grid_pubmedbert.rds"))

# Master scatter frame (full 7470 — subsampled inside Fig 2 only)
sdf <- tibble(
  label = corpus$label,
  disease = DISEASE_SHORT[as.character(corpus$label)],
  pca_tfidf_x = pca2$tfidf[, 1], pca_tfidf_y = pca2$tfidf[, 2],
  pca_pbmed_x = pca2$pubmedbert[, 1], pca_pbmed_y = pca2$pubmedbert[, 2],
  umap_tfidf_x = umap2$tfidf[, 1], umap_tfidf_y = umap2$tfidf[, 2],
  umap_pbmed_x = umap2$pubmedbert[, 1], umap_pbmed_y = umap2$pubmedbert[, 2],
  hdb_cl = as.character(labels_hdb)
)

cat(sprintf(
  "   corpus: %d docs | table3: %d rows | per_class_f1: %d rows\n\n",
  nrow(corpus), nrow(table3), nrow(per_class_f1)
))

# =============================================================================
# Section 1: Figure 2 (PCA vs UMAP Comparison)
# Generates a 2x2 multi-panel layout illustrating the dimensionality reduction
# effectiveness of PCA versus UMAP. Utilizes a stratified subsample of 1,500
# points to maintain optimal visual density and prevent severe overplotting.
# =============================================================================
cat("── PRIMARY FIGURES ──\n")
cat("1. Fig 2 — PCA vs UMAP panel...\n")

set.seed(42)
sdf_sub <- sdf %>%
  group_by(label) %>%
  slice_sample(n = 300) %>%
  ungroup()
stopifnot(nrow(sdf_sub) == 1500L)

# Panel helper — no legend in individual panels; collected at figure level
spanel <- function(df, xc, yc, xl, yl, ttl, sub) {
  ggplot(df, aes(.data[[xc]], .data[[yc]], colour = disease)) +
    geom_point(alpha = 0.35, size = 0.55, stroke = 0) +
    scale_colour_manual(values = DISEASE_COLORS, name = NULL) +
    labs(title = ttl, subtitle = sub, x = xl, y = yl) +
    theme_paper() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_rect(fill = NA, colour = "grey75", linewidth = 0.35)
    )
}

p2a <- spanel(
  sdf_sub, "pca_tfidf_x", "pca_tfidf_y", "PC1", "PC2",
  "TF-IDF  —  PCA", NULL
)
p2b <- spanel(
  sdf_sub, "umap_tfidf_x", "umap_tfidf_y", "UMAP 1", "UMAP 2",
  "TF-IDF  —  UMAP", NULL
)
p2c <- spanel(
  sdf_sub, "pca_pbmed_x", "pca_pbmed_y", "PC1", "PC2",
  "PubMedBERT  —  PCA", NULL
)
p2d <- spanel(
  sdf_sub, "umap_pbmed_x", "umap_pbmed_y", "UMAP 1", "UMAP 2",
  "PubMedBERT  —  UMAP", NULL
)

# patchwork: collect all legends to bottom
p2 <- (p2a | p2b) / (p2c | p2d) +
  plot_layout(guides = "collect") &
  guides(colour = guide_legend(
    nrow = 1,
    byrow = TRUE,
    override.aes = list(alpha = 1, size = 2.8)
  )) &
  theme(legend.position = "bottom", legend.title = element_blank())

save_fig(p2, "fig2_pca_vs_umap", w = 6.5, h = 5.2)

# =============================================================================
# Section 2: Figure 3 (Adjusted Rand Index Comparison)
# Constructs a grouped bar chart visualizing the quantitative performance
# differences (ARI) between representations across three clustering algorithms.
# =============================================================================
cat("2. Fig 3 — ARI bar chart...\n")

fig3 <- table3 %>%
  mutate(
    Representation = factor(Representation, levels = c("TF-IDF", "PubMedBERT")),
    Algorithm = factor(Algorithm,
      levels = c("K-Means", "HDBSCAN", "Agglomerative"),
      labels = c("K-Means", "HDBSCAN\n(k≤20)", "Agglomerative")
    ),
    noise_lbl = ifelse(Noise_pct > 0, paste0(round(Noise_pct), "%\nnoise"), ""),
    ari_lbl = sprintf("%.3f", ARI)
  )

p3 <- ggplot(fig3, aes(Algorithm, ARI, fill = Representation)) +
  geom_col(
    position = position_dodge(0.72), width = 0.65,
    colour = "white", linewidth = 0.2
  ) +
  geom_text(aes(label = ari_lbl, y = ARI + 0.004),
    position = position_dodge(0.72),
    vjust = 0, size = 2.4, colour = "grey25"
  ) +
  geom_text(
    data = filter(fig3, Noise_pct > 0),
    aes(label = noise_lbl, y = ARI / 2),
    position = position_dodge(0.72),
    size = 1.9, colour = "white", fontface = "italic", lineheight = 0.85
  ) +
  scale_fill_manual(values = REP_COL, name = NULL) +
  scale_y_continuous(
    limits = c(0, 0.46), breaks = seq(0, 0.4, 0.1),
    expand = expansion(mult = c(0, 0.06))
  ) +
  labs(
    x = NULL, y = "Adjusted Rand Index"
  ) +
  theme_paper() +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())

save_fig(p3, "fig3_ari_comparison", w = 3.5, h = 3.2)

# =============================================================================
# Section 3: Figure 4 (HDBSCAN Cluster Topography)
# Renders the full 7,470 document corpus projected via UMAP, color-coded by
# the optimal HDBSCAN clustering assignments. Explicitly models noise as a
# foundational background layer to highlight core structural formations.
# =============================================================================
cat("3. Fig 4 — HDBSCAN cluster map...\n")

fig4 <- sdf %>%
  mutate(
    cl_name = ifelse(hdb_cl == "0", "Noise",
      CLUSTER_NAMES[hdb_cl]
    ),
    cl_name = factor(cl_name, levels = names(CLUSTER_COLORS)),
    is_noise = hdb_cl == "0"
  ) %>%
  group_by(cl_name) %>%
  slice_sample(n = 300) %>%
  ungroup() %>%
  arrange(is_noise) # noise plotted first (background layer)

# Centroids for labels — non-noise only
centroids4 <- fig4 %>%
  filter(!is_noise) %>%
  group_by(cl_name) %>%
  summarise(
    cx = median(umap_tfidf_x),
    cy = median(umap_tfidf_y), .groups = "drop"
  )

p4 <- ggplot(fig4, aes(umap_tfidf_x, umap_tfidf_y, colour = cl_name)) +
  geom_point(
    data = filter(fig4, is_noise),
    alpha = 0.45, size = 0.40, stroke = 0
  ) +
  geom_point(
    data = filter(fig4, !is_noise),
    alpha = 0.60, size = 0.55, stroke = 0
  ) +
  geom_label(
    data = centroids4,
    aes(cx, cy, label = cl_name, colour = cl_name),
    fill = alpha("white", 0.85), size = 2.1,
    linewidth = 0.18, fontface = "bold",
    show.legend = FALSE
  ) +
  scale_colour_manual(values = CLUSTER_COLORS, name = NULL) +
  guides(colour = guide_legend(
    nrow = 2, byrow = TRUE,
    override.aes = list(alpha = 1, size = 3)
  )) +
  coord_cartesian(clip = "off") +
  labs(
    x = "UMAP 1", y = "UMAP 2"
  ) +
  theme_paper() +
  theme(
    axis.text = element_blank(), axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, colour = "grey75", linewidth = 0.35),
    legend.position = "bottom",
    legend.text = element_text(size = 7),
    legend.title = element_text(size = 7.5, face = "bold"),
    legend.key.size = unit(0.28, "cm"),
    plot.margin = margin(8, 8, 4, 8)
  )

save_fig(p4, "fig4_hdbscan_cluster_map", w = 5.5, h = 5.2)

# =============================================================================
# Section 4: Figure S1 (K-Means Silhouette Parameter Sweep)
# Visualizes the mean silhouette scores across k in {2...10} to empirically
# justify the optimal selection of k=5 for both representations.
# =============================================================================
cat("4. Fig S1 — K-Means silhouette sweep...\n")

sweep <- bind_rows(
  mutate(sweep_tfidf, Representation = "TF-IDF"),
  mutate(sweep_pbmed, Representation = "PubMedBERT")
) %>%
  mutate(Representation = factor(Representation, levels = c("TF-IDF", "PubMedBERT")))

ps1 <- ggplot(sweep, aes(k, silhouette, colour = Representation, group = Representation)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.2) +
  geom_vline(
    xintercept = 5, linetype = "dashed",
    colour = "grey45", linewidth = 0.5
  ) +
  annotate("text",
    x = 5.18,
    y = min(sweep$silhouette, na.rm = TRUE) + 0.003,
    label = "k = 5", hjust = 0, size = 2.4, colour = "grey40"
  ) +
  scale_colour_manual(values = REP_COL, name = NULL) +
  scale_x_continuous(breaks = 2:10) +
  scale_y_continuous(labels = label_number(accuracy = 0.01)) +
  labs(
    title = "K-Means silhouette sweep  k ∈ {2, …, 10}",
    x = "k", y = "Mean silhouette"
  ) +
  theme_paper() +
  theme(legend.position = "top")

save_fig(ps1, "figs1_kmeans_sweep", w = 3.5, h = 3.2)

# =============================================================================
# Section 5: Figure S2 (HDBSCAN minPts Grid Search)
# Illustrates the parameter search space for density-based clustering, highlighting
# the exclusion zone for degenerate topologies (k > 20) and the selected optimum.
# =============================================================================
cat("5. Fig S2 — HDBSCAN minPts grid...\n")

grid_df <- bind_rows(
  mutate(grid_tfidf, Representation = "TF-IDF"),
  mutate(grid_pbmed, Representation = "PubMedBERT")
) %>%
  mutate(
    Representation = factor(Representation, levels = c("TF-IDF", "PubMedBERT")),
    excluded       = n_clusters > 20,
    selected       = minPts == 100
  )

ps2 <- ggplot(grid_df, aes(minPts, n_clusters,
  colour = Representation, group = Representation
)) +
  # Exclusion zone — ymin=1 avoids log10(0)=-Inf
  annotate("rect",
    xmin = 10, xmax = 55, ymin = 1, ymax = Inf,
    fill = "#E05C4B", alpha = 0.07
  ) +
  annotate("text",
    x = 19, y = 3.2, label = "k>20\nexcluded",
    size = 2.0, colour = "#B03020", fontface = "italic", lineheight = 0.85
  ) +
  geom_line(linewidth = 0.75) +
  geom_point(aes(shape = excluded), size = 2.5, stroke = 0.9) +
  geom_point(
    data = filter(grid_df, selected),
    shape = 21, size = 5.5, fill = NA,
    colour = "black", stroke = 1.0, show.legend = FALSE
  ) +
  scale_colour_manual(values = REP_COL, name = NULL) +
  scale_shape_manual(
    values = c("FALSE" = 16, "TRUE" = 4),
    labels = c("FALSE" = "k≤20", "TRUE" = "k>20"),
    name   = NULL
  ) +
  scale_x_log10(
    breaks = c(10, 25, 50, 100, 200, 500),
    labels = label_number(accuracy = 1)
  ) +
  scale_y_log10(
    breaks = c(2, 6, 10, 25, 50, 100, 200),
    labels = label_number(accuracy = 1)
  ) +
  labs(
    title = "HDBSCAN minPts grid search",
    subtitle = "Circle = selected config (minPts=100, noise≈30%)",
    x = "minPts  (log)", y = "k  (log)"
  ) +
  guides(
    colour = guide_legend(order = 1),
    shape  = guide_legend(order = 2)
  ) +
  theme_paper() +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.spacing.x = unit(0.25, "cm")
  )

save_fig(ps2, "figs2_hdbscan_grid", w = 3.5, h = 3.5)

# =============================================================================
# Section 6: Figure 5 (Cluster-to-Disease Confusion Matrix)
# Formats a row-normalized contingency heatmap detailing the proportional
# composition of each derived cluster relative to ground-truth disease classes.
# Evaluated strictly on the best-performing experimental pipeline.
# =============================================================================
cat("\n── ADDITIONAL FIGURES ──\n")
cat("6. Fig 5 — Confusion matrix...\n")

non_noise <- labels_hdb != 0
cm_pred_raw <- as.character(labels_hdb[non_noise])
cm_true_raw <- as.character(corpus$label[non_noise])

cm_pred <- factor(CLUSTER_NAMES[cm_pred_raw],
  levels = c(
    "Neoplasms", "Digestive", "IBD", "Gen. Path.",
    "Cardiovascular", "Nervous", "Neurodegen.", "Palliative"
  )
)
cm_true <- factor(DISEASE_SHORT[cm_true_raw],
  levels = c(
    "Neoplasms", "Digestive", "Nervous",
    "Cardiovascular", "Gen. Path."
  )
)

ct_raw <- table(Cluster = cm_pred, Disease = cm_true)
# Row-normalise: % of each cluster's documents
ct_pct <- round(prop.table(ct_raw, margin = 1) * 100, 1)

cm_df <- as.data.frame(ct_pct) %>%
  rename(pct = Freq)

p5 <- ggplot(cm_df, aes(Disease, Cluster, fill = pct)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(
    aes(
      label = ifelse(pct >= 4, paste0(pct, "%"), ""),
      colour = pct > 45
    ),
    size = 2.6, fontface = "bold", show.legend = FALSE
  ) +
  scale_colour_manual(values = c("TRUE" = "white", "FALSE" = "grey20")) +
  scale_fill_gradient(
    low = "#F0F4F8", high = "#102A43",
    name = "% of cluster", limits = c(0, 100)
  ) +
  labs(
    x = NULL, y = NULL
  ) +
  theme_paper() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 7.5),
    axis.text.y = element_text(size = 7.5),
    legend.position = "right",
    panel.grid = element_blank()
  )

save_fig(p5, "fig5_confusion_matrix", w = 5.0, h = 3.8)

# =============================================================================
# Section 7: Figure 8 (Per-Class F1 Score Heatmap)
# Provides a comprehensive micro-evaluation comparing the F1 accuracy of all
# experimental combinations stratified by individual disease categories.
# =============================================================================
cat("8. Fig 8 — Per-class F1 heatmap...\n")

f1_hm <- per_class_f1 %>%
  mutate(
    exp_label = case_when(
      experiment == "tfidf_kmeans" ~ "TF-IDF\nK-Means",
      experiment == "tfidf_hdbscan" ~ "TF-IDF\nHDBSCAN",
      experiment == "tfidf_agglomerative" ~ "TF-IDF\nAgglom.",
      experiment == "pubmedbert_kmeans" ~ "PubMedBERT\nK-Means",
      experiment == "pubmedbert_hdbscan" ~ "PubMedBERT\nHDBSCAN",
      experiment == "pubmedbert_agglomerative" ~ "PubMedBERT\nAgglom."
    ),
    exp_label = factor(exp_label, levels = c(
      "TF-IDF\nK-Means", "TF-IDF\nHDBSCAN", "TF-IDF\nAgglom.",
      "PubMedBERT\nK-Means", "PubMedBERT\nHDBSCAN", "PubMedBERT\nAgglom."
    )),
    dis_short = case_when(
      disease_name == "Neoplasms" ~ "Neoplasms",
      disease_name == "Digestive System Diseases" ~ "Digestive",
      disease_name == "Nervous System Diseases" ~ "Nervous",
      disease_name == "Cardiovascular Diseases" ~ "Cardiovascular",
      disease_name == "General Pathological Conditions" ~ "Gen. Path."
    ),
    dis_short = factor(dis_short,
      levels = rev(c(
        "Cardiovascular", "Neoplasms",
        "Digestive", "Nervous", "Gen. Path."
      ))
    )
  )

p8 <- ggplot(f1_hm, aes(exp_label, dis_short, fill = f1)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", f1)),
    size = 2.5, colour = "grey15"
  ) +
  # Vertical separator between TF-IDF and PubMedBERT groups
  geom_segment(
    x = 3.5, xend = 3.5, y = 0.5, yend = 5.5, colour = "grey50", linewidth = 0.6,
    linetype = "dashed"
  ) +
  scale_fill_gradient(
    low = "grey96", high = "#E84040",
    name = "F1", limits = c(0, 1)
  ) +
  scale_x_discrete(position = "top") +
  coord_cartesian(clip = "off") +
  # Annotation for repr groups
  annotate("text",
    x = 2, y = -0.15, label = "TF-IDF", size = 2.8,
    colour = REP_COL["TF-IDF"], fontface = "bold"
  ) +
  annotate("text",
    x = 4.9, y = -0.15, label = "PubMedBERT", size = 2.8,
    colour = REP_COL["PubMedBERT"], fontface = "bold"
  ) +
  labs(
    x = NULL, y = NULL
  ) +
  theme_paper() +
  theme(
    axis.text.x       = element_text(size = 5.0, lineheight = 0.65, margin = margin(b = -2)),
    axis.text.y       = element_text(size = 6.5, margin = margin(r = -2)),
    axis.ticks        = element_blank(),
    axis.ticks.x      = element_blank(),
    axis.ticks.y      = element_blank(),
    axis.ticks.length = unit(0, "pt"),
    panel.border      = element_blank(),
    panel.grid        = element_blank(),
    panel.grid.major  = element_blank(),
    panel.grid.minor  = element_blank(),
    legend.position   = "right",
    legend.key.height = unit(0.7, "cm"),
    legend.key.width  = unit(0.25, "cm"),
    plot.margin       = margin(4, 4, 18, 4)
  )

save_fig(p8, "fig8_per_class_f1", w = 5.2, h = 2.8)

# =============================================================================
# Section 8: Pipeline Completion and Validation Logging
# =============================================================================
cat(sprintf("\n── 07_figures.R complete (%.1f sec)\n", elapsed(t0)))
cat("\n   PRIMARY:\n")
cat("   fig2_pca_vs_umap          6.5 × 5.2\n")
cat("   fig3_ari_comparison       3.5 × 3.2\n")
cat("   fig4_hdbscan_cluster_map  5.5 × 5.2\n")
cat("   figs1_kmeans_sweep        3.5 × 3.2\n")
cat("   figs2_hdbscan_grid        3.5 × 3.5\n")
cat("\n   ADDITIONAL:\n")
cat("   fig5_confusion_matrix     5.0 × 3.8\n")
cat("   fig8_per_class_f1         5.2 × 2.8\n")
cat("\n   All figures: PDF (vector) + PNG (300 dpi)\n")
cat("   Pipeline complete.\n")
