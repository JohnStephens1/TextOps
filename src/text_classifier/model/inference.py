import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    EMBEDDING_MODEL_STR_ARTIFACT_PATH,
    ENCODER_ARTIFACT_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.features import add_features
from text_classifier.data.model_data import drop_non_feature_cols
from text_classifier.data.preprocessing import preprocess_features
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_joblib, load_text

setup_logging()


logger = logging.getLogger("text_classifier.model.inference")


# def fancy_schmancy_sort_key(col: str) -> tuple[int, int | str]:
#     """sorts columns alphabetically, aside from text_<int> columns, which come last. text_<int> columns are sorted numerically, based on their <int>."""
#     splits = col.split("_")

#     if splits[0] == "text" and splits[-1].isdigit():
#         return (1, int(splits[-1]))
#     else:
#         return (0, col)


def get_df_from_input(title: str, description: str, date_time: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "created_on": [date_time],
            "title": [title],
            "description": [description],
        }
    )


def get_current_date_time() -> datetime:
    return datetime.now().astimezone()


def raw_to_model_input_pipe(
    title: str, description: str, date_time: str
) -> pd.DataFrame:
    df = get_df_from_input(title, description, date_time)
    df = preprocess_features(df)
    df = add_features(df, regenerate_embs=True)
    df = drop_non_feature_cols(df)

    return df


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


def get_champ_model_encoder_emb_model_str() -> tuple[Predictor, LabelEncoder, str]:
    client = mlflow.MlflowClient("http://localhost:5000")

    run_id = get_champion_run_id(client)
    label_encoder, embedding_model_str = get_label_encoder_emb_model_str(client, run_id)
    model = get_champion_model()

    return model, label_encoder, embedding_model_str


# date_time: str | datetime ?
def inference(
    title: str, description: str, date_time: str
) -> tuple[np.ndarray, np.ndarray]:
    # TODO add pydantic check

    model, label_encoder, _ = get_champ_model_encoder_emb_model_str()
    df = raw_to_model_input_pipe(title, description, date_time)

    preds_proba = model.predict_proba(df)
    preds = model.predict(df)
    labels = label_encoder.inverse_transform(preds)

    print(f"""
        Input:
        - title: {title}
        - description: {description}
        Output:
        - possible labels: {label_encoder.classes_}
        - certainty: {[f"{x:.4f}" for x in preds_proba[0]]}
        - prediction: {labels}
    """)

    return preds_proba, labels


# print(inference("langue", "fabricating and refining", str(get_current_date_time())))
