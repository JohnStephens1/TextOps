from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


CONFIG_PATH = PROJECT_ROOT / "config"


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
TRAIN_RUN_ID_PATH = TRAIN_ARTIFACTS_DIR / "run_id.txt"
TRAIN_MODEL_VERSION_PATH = TRAIN_ARTIFACTS_DIR / "model_version.txt"


EVAL_DIR = ARTIFACTS_DIR / "eval"

PLOTS_DIR = EVAL_DIR / "plots"
FIGS_DIR = EVAL_DIR / "figs"
METRICS_DIR = EVAL_DIR / "metrics"

TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"


MODEL_DIR = PROJECT_ROOT / "models"

PRODUCTION_MODEL_DIR = MODEL_DIR / "production"
CANDIDATE_MODEL_DIR = MODEL_DIR / "candidate"

PRODUCTION_MODEL_PATH = PRODUCTION_MODEL_DIR / "model.joblib"
PRODUCTION_MODEL_METRICS_PATH = PRODUCTION_MODEL_DIR / "metrics.json"


EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/text_embeddings.npz"
TRAIN_METADATA_PATH = PROJECT_ROOT / "artifacts/train_metadata.json"

EMBEDDING_MODEL_STR = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
