"""
=============================================================================
embed_pubmedbert.py — PubMedBERT Contextual Embedding Generation
BioCluster-R | ICCA 2026

Generates dense 768-dimensional contextual embeddings for Track B raw texts.
Enforces strict single-threaded CPU execution to guarantee floating-point 
determinism and absolute reproducibility across hardware architectures.
Exports to Apache Parquet for efficient ingestion into the R DR pipeline.
=============================================================================
"""
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =============================================================================
# Section 1: Hardware Determinism Configuration
# Restricts PyTorch to a single thread to guarantee exact computational 
# reproducibility, circumventing floating-point variations inherent to multi-threading.
# =============================================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# =============================================================================
# Section 2: Directory and Path Initialization
# =============================================================================
ROOT = Path(__file__).parent.parent
INPUT  = ROOT / "data/processed/track_b_for_python.csv"
OUTPUT = ROOT / "data/processed/embeddings_pubmedbert.parquet"
MODEL_CACHE = ROOT / "python/.model_cache"

# =============================================================================
# Section 3: Data Import and Validation (Track B)
# Loads raw, un-lemmatized texts designed for internal WordPiece tokenization.
# =============================================================================
df = pd.read_csv(INPUT)
assert list(df.columns) == ["doc_id", "label", "text_raw"], \
    "Column mismatch — regenerate track_b_for_python.csv"
print(f"Loaded {len(df)} documents.")

# =============================================================================
# Section 4: Deep Learning Model Initialization
# Initializes the NeuML/pubmedbert-base-embeddings model from the Hugging Face 
# cache. Enforces explicit CPU device allocation to prevent non-deterministic GPU operations.
# =============================================================================
model = SentenceTransformer(
    "NeuML/pubmedbert-base-embeddings",
    cache_folder=str(MODEL_CACHE),
    device="cpu"          # explicit CPU — never MPS for determinism
)

# =============================================================================
# Section 5: Embedding Computation and Normalization
# Derives 768-dimensional sentence embeddings. The L2 normalization step is 
# applied directly during encoding to yield unit-length vectors.
# =============================================================================
embeddings = model.encode(
    df["text_raw"].tolist(),
    batch_size=64,
    normalize_embeddings=True,  # L2-norm: unit vectors
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Embedding shape: {embeddings.shape}")  # (7470, 768)

# =============================================================================
# Section 6: Vector Norm Quality Assurance
# Empirically validates that all generated embeddings successfully achieve L2 
# unit-length, an absolute requisite for downstream cosine-based distance calculations.
# =============================================================================
import numpy as np
norms = np.linalg.norm(embeddings, axis=1)
assert np.all(np.abs(norms - 1.0) < 0.001), "L2 norm check failed"
print(f"L2 norm: min={norms.min():.6f} max={norms.max():.6f}  ✓")

# =============================================================================
# Section 7: Dimensional Output Structuring
# =============================================================================
dim_cols = {f"dim_{i}": embeddings[:, i] for i in range(768)}
out_df = pd.DataFrame({"doc_id": df["doc_id"], "label": df["label"], **dim_cols})

# =============================================================================
# Section 8: High-Performance Parquet Serialization
# Exports the final embedding matrix using Snappy compression for optimal I/O 
# integration with the R representation pipeline.
# =============================================================================
table = pa.Table.from_pandas(out_df, preserve_index=False)
pq.write_table(table, OUTPUT, compression="snappy")
print(f"Saved: {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")