import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    EMBEDDING_MODEL_STR_ARTIFACT_PATH,
    ENCODER_ARTIFACT_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_joblib, load_text

setup_logging()


logger = logging.getLogger("text_classifier.mlflow_loader")


def get_mlflow_artifact(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
    fallback_path: Path,
    loader: Callable[[Path], Any],
) -> Any:
    try:
        path = Path(
            client.download_artifacts(
                run_id=run_id,
                path=mlflow_path,
                dst_path=tmp_dir,
            )
        )
    except mlflow.exceptions.MlflowException:  # type: ignore
        logger.warning(
            f"Champion artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )
        path = fallback_path

    return loader(path)


def get_label_encoder_emb_model_str(
    client: mlflow.MlflowClient, run_id: str
) -> tuple[LabelEncoder, str]:
    mlflow_label_encoder_path = "preprocessing/label_encoder.joblib"
    mlflow_embedding_model_str_path = "embeddings/embedding_model_str.txt"

    with TemporaryDirectory() as tmp_dir:
        label_encoder: LabelEncoder = get_mlflow_artifact(
            client,
            run_id,
            mlflow_label_encoder_path,
            tmp_dir,
            ENCODER_ARTIFACT_PATH,
            load_joblib,
        )

        embedding_model_str: str = get_mlflow_artifact(
            client,
            run_id,
            mlflow_embedding_model_str_path,
            tmp_dir,
            EMBEDDING_MODEL_STR_ARTIFACT_PATH,
            load_text,
        )

    return label_encoder, embedding_model_str


def get_champion_model() -> Predictor:
    mlflow.set_tracking_uri("http://localhost:5000")

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


def get_champ_model_encoder_emb_model() -> tuple[
    Predictor, LabelEncoder, SentenceTransformer
]:
    client = mlflow.MlflowClient("http://localhost:5000")

    run_id = get_champion_run_id(client)
    label_encoder, embedding_model_str = get_label_encoder_emb_model_str(client, run_id)
    embedding_model = SentenceTransformer(embedding_model_str)
    model = get_champion_model()

    return model, label_encoder, embedding_model
