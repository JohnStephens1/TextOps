import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from common.environment import MLFLOW_URL
from common.logging_config import setup_logging
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    EMBEDDING_MODEL_STR,
    ENCODER_ARTIFACT_PATH,
)
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_joblib, load_text
from text_classifier.schema import PredictionResources

setup_logging()

logger = logging.getLogger("text_classifier.mlflow_loader")


def load_label_encoder(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
) -> LabelEncoder:
    try:
        path = Path(
            client.download_artifacts(
                run_id=run_id,
                path=mlflow_path,
                dst_path=tmp_dir,
            )
        )
    except mlflow.exceptions.MlflowException as e:  # type: ignore
        logger.warning(f"MLFlow exception occurred: {e}")
        logger.warning(
            f"Champion label encoder artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )
        path = ENCODER_ARTIFACT_PATH

    return load_joblib(path)


def load_embedding_model(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
) -> SentenceTransformer:
    try:
        path = Path(
            client.download_artifacts(
                run_id=run_id,
                path=mlflow_path,
                dst_path=tmp_dir,
            )
        )
        model_str = load_text(path)
    except mlflow.exceptions.MlflowException as e:  # type: ignore
        logger.warning(f"MLFlow exception occurred: {e}")
        logger.warning(
            f"Champion embedding model artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )
        model_str = EMBEDDING_MODEL_STR

    return SentenceTransformer(model_str)


def get_label_encoder_emb_model_str(
    client: mlflow.MlflowClient, run_id: str
) -> tuple[LabelEncoder, SentenceTransformer]:
    mlflow_label_encoder_path = "preprocessing/label_encoder.joblib"
    mlflow_embedding_model_str_path = "embeddings/embedding_model_str.txt"

    with TemporaryDirectory() as tmp_dir:
        label_encoder: LabelEncoder = load_label_encoder(
            client,
            run_id,
            mlflow_label_encoder_path,
            tmp_dir,
        )

        embedding_model_str = load_embedding_model(
            client,
            run_id,
            mlflow_embedding_model_str_path,
            tmp_dir,
        )

    return label_encoder, embedding_model_str


def get_champion_model() -> Predictor:
    mlflow.set_tracking_uri(MLFLOW_URL)

    model: Predictor | None = mlflow.sklearn.load_model(  # type: ignore
        "models:/text_classifier@champion"
    )

    if model is None:
        raise ValueError("Champion model not found.")

    return model


def get_champion_run_id(
    client: mlflow.MlflowClient,
) -> str:
    champion_version = client.get_model_version_by_alias("text_classifier", "champion")

    if champion_version.run_id is None:
        raise AttributeError("Champion run id not found.")

    return champion_version.run_id


def get_champ_model_prediction_resources() -> PredictionResources:
    client = mlflow.MlflowClient(MLFLOW_URL)

    run_id = get_champion_run_id(client)

    label_encoder, embedding_model = get_label_encoder_emb_model_str(client, run_id)
    model = get_champion_model()

    return PredictionResources(model, label_encoder, embedding_model)
