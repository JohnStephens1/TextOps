from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS_DIR = PROJECT_ROOT / "data/datasets"
RAW_DATASET_PATH = DATASETS_DIR / "raw_dataset.csv"
PREPROCESSED_DATASET_PATH = DATASETS_DIR / "preprocessed_dataset.parquet"

EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/text_embeddings.npz"

EMBEDDING_MODEL_STR = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
