import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    EMBEDDING_MODEL_STR,
    ENCODER_ARTIFACT_PATH,
)
from text_classifier.config.environment import MLFLOW_URL
from text_classifier.config.logging_config import setup_logging
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_joblib, load_text
from text_classifier.schema import PredictionResources

setup_logging()

logger = logging.getLogger("text_classifier.mlflow_loader")


def get_mlflow_artifact(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
    fallback_path: Path | str,
    loader: Callable[[Path], Any] | type[SentenceTransformer],
) -> Any:
    # mlflow.set_tracking_uri(MLFLOW_URL)

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
            f"Champion artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )
        path = fallback_path
        return EMBEDDING_MODEL_STR

    return loader(path)


def load_label_encoder(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
) -> Any:
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
            f"Champion artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )
        path = ENCODER_ARTIFACT_PATH

    return load_joblib(path)


def load_embedding_model(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
) -> Any:
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
            f"Champion artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )
        model_str = EMBEDDING_MODEL_STR

    return SentenceTransformer(model_str)


def get_mlflow_artifact_path(
    client: mlflow.MlflowClient,
    run_id: str,
    mlflow_path: str,
    tmp_dir: str,
) -> Path | None:
    try:
        return Path(
            client.download_artifacts(
                run_id=run_id,
                path=mlflow_path,
                dst_path=tmp_dir,
            )
        )
    except mlflow.exceptions.MlflowException as e:  # type: ignore
        logger.warning(f"MLFlow exception occurred: {e}")
        logger.warning(
            f"Champion artifact not found in MLFlow: {mlflow_path}\nTrying to default to local alternative..."
        )

        return None


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
            Path(EMBEDDING_MODEL_STR),
            str,
        )

        # label_encoder_path = get_mlflow_artifact_path(
        #     client,
        #     run_id,
        #     mlflow_label_encoder_path,
        #     tmp_dir,
        # ) or ENCODER_ARTIFACT_PATH

        # label_encoder: LabelEncoder = load_joblib(label_encoder_path)

        # embedding_model_path = get_mlflow_artifact_path(
        #     client,
        #     run_id,
        #     mlflow_embedding_model_str_path,
        #     tmp_dir,
        # )

        # if embedding_model_path is None:
        #     embedding_model_str = EMBEDDING_MODEL_STR
        # else:
        #     embedding_model_str = load_text(embedding_model_path)

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

    label_encoder, embedding_model_str = get_label_encoder_emb_model_str(client, run_id)
    embedding_model = SentenceTransformer(embedding_model_str)
    model = get_champion_model()

    return PredictionResources(model, label_encoder, embedding_model)
