import joblib  # type: ignore
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    ENCODER_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    RUN_ID_ARTIFACT_PATH,
)


def save_model_encoder_run_id(model: BaseEstimator, encoder: LabelEncoder, run_id: str):
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    joblib.dump(encoder, ENCODER_ARTIFACT_PATH)

    RUN_ID_ARTIFACT_PATH.write_text(run_id, encoding="utf-8")


def load_model_encoder_run_id() -> tuple[BaseEstimator, LabelEncoder, str]:
    model = joblib.load(MODEL_ARTIFACT_PATH)
    encoder = joblib.load(ENCODER_ARTIFACT_PATH)

    run_id = RUN_ID_ARTIFACT_PATH.read_text(encoding="utf-8")

    return model, encoder, run_id
