# BioCluster-R: A Reproducible Pipeline for Biomedical Text Clustering

This repository contains the complete, production-ready reproducibility pipeline for the paper *"BioCluster: A Comparative Analysis of Contextual and Traditional Representations in Unsupervised Biomedical Text Clustering"*. It provides a rigorous framework for evaluating traditional (TF-IDF) versus deep contextual (PubMedBERT) representations in unsupervised clustering of biomedical abstracts.

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Acquisition](#data-acquisition)
- [Environment Setup](#environment-setup)
- [Execution & Reproducibility](#execution--reproducibility)
- [Hardware Determinism](#hardware-determinism)

## 🔬 Overview
BioCluster-R is a highly optimized, reproducible R and Python pipeline that processes medical text corpora, extracts representations, and performs rigorous clustering evaluation (K-Means, HDBSCAN, Agglomerative). It integrates the Hungarian algorithm for optimal cluster-to-class alignment, calculates multiple evaluation metrics (ARI, Macro-F1, V-Measure), and conducts permutation testing to determine algorithmic superiority.

## 📂 Repository Structure
```text
BioCluster-R/
├── R/                          # R scripts for preprocessing, clustering, and evaluation
│   ├── 00_eda.R                # Exploratory Data Analysis
│   ├── 01_preprocessing.R      # 7-Step Text Preprocessing Pipeline
│   ├── 02_representations.R    # TF-IDF & Embedding Initialization
│   ├── 03_umap_pca.R           # Dimensionality Reduction (UMAP/PCA)
│   ├── 04_clustering.R         # K-Means, HDBSCAN, Agglomerative Clustering
│   ├── 05_evaluation.R         # Metric Computation & Significance Testing
│   ├── 06_keywords.R           # c-TF-IDF Keyword Extraction
│   ├── 07_figures.R            # Publication-ready Figure Generation
│   └── utils.R                 # Global path configurations and utility functions
├── python/
│   ├── embed_pubmedbert.py     # PubMedBERT contextual embedding generation
│   └── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                    # Raw dataset files and udpipe models
│   ├── processed/              # Cleaned text and baseline stats
│   ├── embeddings/             # Pre-computed parquet embeddings
│   ├── repr/                   # Representation matrices (generated)
│   ├── umap/                   # Dimensionality reduction artifacts (generated)
│   └── clustering/             # Clustering label artifacts (generated)
├── results/                    # Output evaluation tables & metrics (generated)
├── figures/                    # Output publication figures (generated)
├── renv.lock                   # Exact R environment lockfile
└── README.md
```

## 📥 Data Acquisition

### 1. Medical Abstracts Dataset
The raw dataset is sourced from the Medical Abstracts TC Corpus.
1. Download `medical_tc_train.csv` and `medical_tc_test.csv` from the [official GitHub repository](https://github.com/sebischair/Medical-Abstracts-TC-Corpus/tree/main).
2. Place both files in the `data/raw/` directory.

### 2. UDPipe Language Model
The text preprocessing pipeline relies on the English EWT UD model for lemmatization.
1. Download the `english-ewt-ud-2.5-191206.udpipe` model.
2. Place the `.udpipe` file directly in the project root (or inside the `data/raw/` directory).

*(Note: The `data/processed/` directory in this repository already contains the balanced corpus (`corpus_balanced.rds`), and the `data/embeddings/` directory contains the pre-computed embeddings for immediate reproduction.)*

## 🛠 Environment Setup
This project strictly enforces dependency versions to guarantee bit-identical reproducibility.

### R Environment
Ensure you have R installed (>= 4.1.0). From the project root, open R/RStudio and restore the environment:
```R
# Install renv if not already present
install.packages("renv")

# Restore the exact package environment from renv.lock
renv::restore()
```

### Python Environment
The deep learning embedding extraction step requires Python (>= 3.9). Create a virtual environment and install the required dependencies:
```bash
cd python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Execution & Reproducibility
The pipeline is meticulously designed to be executed sequentially. All path routing is dynamically handled by `utils.R`.

### Step 1: Preprocessing & Data Preparation
Run the initial R scripts to establish the baseline datasets.
```R
source("R/00_eda.R")
source("R/01_preprocessing.R")
```

### Step 2: Contextual Embeddings Generation
You have two options for handling the PubMedBERT contextual embeddings:
*   **Option A (Fast Track):** Use the pre-computed `embeddings_pubmedbert.parquet` file already provided in `data/embeddings/`. You may proceed directly to Step 3.
*   **Option B (Full Reproduction):** Run the Python script to generate them from scratch.
    ```bash
    source python/.venv/bin/activate
    python python/embed_pubmedbert.py
    ```

### Step 3: Clustering, Evaluation, and Visualization
Execute the remaining R scripts in sequential order. All figures, tables, and metrics from the ICCA 2026 paper will be exported directly into the `figures/` and `results/` directories.
```R
source("R/02_representations.R")
source("R/03_umap_pca.R")
source("R/04_clustering.R")
source("R/05_evaluation.R")
source("R/06_keywords.R")
source("R/07_figures.R")
```

## ⚙️ Hardware Determinism
To achieve absolute reproducibility and circumvent hardware-specific floating-point arithmetic variations (especially across varying GPU architectures), the following constraints are hardcoded into the pipeline:
*   **Single-Threaded Execution:** UMAP dimensionality reduction (`uwot::umap(n_threads = 1L)`) and PyTorch embedding generation (`torch.set_num_threads(1)`) are strictly constrained to a single CPU core.
*   **Deterministic Seeding:** Global seeds (`set.seed(42)`) are explicitly instantiated prior to any stochastic operation.