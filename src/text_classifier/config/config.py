from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


DATASETS_DIR = PROJECT_ROOT / "data/datasets"

RAW_DATASET_PATH = DATASETS_DIR / "raw_dataset.csv"
PREPROCESSED_DATASET_PATH = DATASETS_DIR / "preprocessed_dataset.parquet"
FEATURE_DATASET_PATH = DATASETS_DIR / "feature_dataset.parquet"
MODEL_DATASET_PATH = DATASETS_DIR / "model_dataset.parquet"

SPLITS_DIR = DATASETS_DIR / "splits"

X_TRAIN_PATH = SPLITS_DIR / "X_train.parquet"
X_TEST_PATH = SPLITS_DIR / "X_test.parquet"
Y_TRAIN_PATH = SPLITS_DIR / "y_train.parquet"
Y_TEST_PATH = SPLITS_DIR / "y_test.parquet"


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "model.joblib"
ENCODER_ARTIFACT_PATH = ARTIFACTS_DIR / "label_encoder.joblib"
RUN_ID_ARTIFACT_PATH = ARTIFACTS_DIR / "run_id.txt"


EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/text_embeddings.npz"
TRAIN_METADATA_PATH = PROJECT_ROOT / "artifacts/train_metadata.json"

EMBEDDING_MODEL_STR = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
