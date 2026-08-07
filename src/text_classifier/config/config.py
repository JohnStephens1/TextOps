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

ENCODER_ARTIFACT_PATH = ARTIFACTS_DIR / "label_encoder.joblib"


TRAIN_ARTIFACTS_DIR = ARTIFACTS_DIR / "train"

TRAIN_BEST_ESTIMATOR_PATH = TRAIN_ARTIFACTS_DIR / "best_estimator.joblib"
TRAIN_BEST_PARAMS_PATH = TRAIN_ARTIFACTS_DIR / "best_params.json"
TRAIN_BEST_SCORE_PATH = TRAIN_ARTIFACTS_DIR / "best_score.txt"
TRAIN_CV_RESULTS_PATH = TRAIN_ARTIFACTS_DIR / "cv_results.parquet"
TRAIN_RUN_ID = TRAIN_ARTIFACTS_DIR / "run_id.txt"


EVAL_DIR = ARTIFACTS_DIR / "eval"

METRICS_DIR = EVAL_DIR / "metrics"
PLOTS_DIR = EVAL_DIR / "plots"
FIGS_DIR = EVAL_DIR / "figs"


EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/text_embeddings.npz"
TRAIN_METADATA_PATH = PROJECT_ROOT / "artifacts/train_metadata.json"

EMBEDDING_MODEL_STR = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
