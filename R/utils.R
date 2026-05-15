# =============================================================================
# utils.R — Shared Constants, Paths, and Helper Functions
# BioCluster-R | ICCA 2026
#
# Purpose:
#   This script serves as the global configuration file sourced by every other 
#   script in the pipeline. It defines structural constants, dynamically creates
#   required directory structures to ensure reproducibility on any machine, 
#   and holds utility functions used universally.
#
# Output:
#   Loads libraries, instantiates the 'PATHS' list, 'DISEASE_MAP', and 
#   'REPR_NAMES' into the global environment. Provides the 'save_artifact' and 
#   'elapsed' functions.
# =============================================================================

library(dplyr)
library(stringr)

# -----------------------------------------------------------------------------
# PROJECT PATH CONFIGURATION
# 
# Defines the absolute hierarchy for the pipeline. By isolating all I/O paths 
# here, the pipeline avoids hardcoded relative directories scattered across 
# multiple scripts.
# -----------------------------------------------------------------------------
PATHS <- list(
  raw        = "data/raw",
  processed  = "data/processed",
  embeddings = "data/embeddings",
  repr       = "data/repr",
  umap       = "data/umap",
  clustering = "data/clustering",
  results    = "results",
  tables     = "tables",
  figures    = "figures"
)

# -----------------------------------------------------------------------------
# DIRECTORY INITIALIZATION
# 
# Iterates through the predefined PATHS list and ensures that every directory 
# exists on the filesystem. This is critical for preventing "No such file or 
# directory" errors on initial run.
# Parameters:
#   recursive = TRUE: creates parent directories if needed.
#   showWarnings = FALSE: silently ignores if the directories already exist.
# -----------------------------------------------------------------------------
invisible(lapply(PATHS, \(p) dir.create(p, recursive = TRUE, showWarnings = FALSE)))

# -----------------------------------------------------------------------------
# DATASET CONSTANTS
# 
# DISEASE_MAP: A named vector mapping the original dataset's integer labels 
# (1-5) to their human-readable, biological disease categories from Schopf et al. 
# 2022 (Medical Abstracts TC Corpus).
# -----------------------------------------------------------------------------
DISEASE_MAP <- c(
  "1" = "Neoplasms",
  "2" = "Digestive System Diseases",
  "3" = "Nervous System Diseases",
  "4" = "Cardiovascular Diseases",
  "5" = "General Pathological Conditions"
)

# -----------------------------------------------------------------------------
# EXPERIMENTAL CONSTANTS
# 
# REPR_NAMES: Keys used for the two primary text representation methodologies 
# being compared in the study. These act as dictionary keys in downstream lists.
# -----------------------------------------------------------------------------
REPR_NAMES <- c("tfidf", "pubmedbert")

# -----------------------------------------------------------------------------
# HELPER FUNCTION: save_artifact
# 
# Purpose:
#   Standardizes data persistence across the pipeline. It automatically saves 
#   any R object to a compressed RDS file. For objects that are data frames, it 
#   also dumps a raw CSV file for easy human inspection and cross-language use.
# 
# Parameters:
#   obj: The R object to save.
#   path_no_ext: The destination path (without file extension).
# 
# Returns:
#   Invisibly returns the object itself.
# -----------------------------------------------------------------------------
save_artifact <- function(obj, path_no_ext) {
  saveRDS(obj, paste0(path_no_ext, ".rds"))
  if (is.data.frame(obj)) {
    write.csv(obj, paste0(path_no_ext, ".csv"), row.names = FALSE)
  }
  invisible(obj)
}

# -----------------------------------------------------------------------------
# HELPER FUNCTION: elapsed
# 
# Purpose:
#   Calculates the time duration in seconds since a captured 'proc.time()'.
#   Used universally for performance logging and benchmarking code execution.
# 
# Parameters:
#   t0: A starting proc.time() object.
# 
# Returns:
#   Numeric scalar of elapsed time rounded to 1 decimal place.
# -----------------------------------------------------------------------------
elapsed <- function(t0) round((proc.time() - t0)[3], 1)
