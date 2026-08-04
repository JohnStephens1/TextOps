import json
import logging
import tempfile
from pathlib import Path

import joblib  # type: ignore
import mlflow
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    ENCODER_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    RUN_ID_ARTIFACT_PATH,
    TRAIN_METADATA_PATH,
)
from text_classifier.data.train_data import get_encoder_train_data
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.evaluation.plots import get_model_eval_figs
from text_classifier.model.models import (
    ModelBase,
    get_model_from_config,
)
from text_classifier.model.predictions import get_predictions_w_encoder
from text_classifier.schema import TrainingData

logger = logging.getLogger(__name__)


def save_model_encoder_run_id(model: BaseEstimator, encoder: LabelEncoder, run_id: str):
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    joblib.dump(encoder, ENCODER_ARTIFACT_PATH)

    RUN_ID_ARTIFACT_PATH.write_text(run_id, encoding="utf-8")


def log_encoder(
    encoder: LabelEncoder,
    encoder_file_name: str = "label_encoder.joblib",
    artifact_dir_name: str = "preprocessing",
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder_path = Path(tmp_dir) / encoder_file_name

        joblib.dump(encoder, encoder_path)
        mlflow.log_artifact(str(encoder_path), artifact_path=artifact_dir_name)


def log_metrics_figs(metrics: dict[str, float], figs: dict[str, Figure]) -> None:
    mlflow.log_metrics(metrics)

    for fig_name, fig in figs.items():
        mlflow.log_figure(fig, f"figures/{fig_name}.png")


def log_train_metadata(run: mlflow.ActiveRun) -> None:
    metadata = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "artifact_uri": run.info.artifact_uri,
        "model_uri": f"runs:/{run.info.run_id}/model",
    }

    with open(TRAIN_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def get_metrics_figs(
    my_model: ModelBase, train_data: TrainingData, encoder: LabelEncoder
) -> tuple[dict[str, float], dict[str, Figure]]:
    predictions_w_encoder = get_predictions_w_encoder(
        my_model.search, train_data.X_test, train_data.y_test, encoder
    )

    metrics = get_classification_metrics(
        predictions_w_encoder.predictions,
        prefix="test",
    )

    figs = get_model_eval_figs(predictions_w_encoder)

    return metrics, figs


def train_core(
    train_data: TrainingData,
    my_model: ModelBase,
) -> tuple[TrainingData, ModelBase]:
    my_model.search.fit(train_data.X_train, train_data.y_train)

    return train_data, my_model


def train_w_tracking(
    train_data: TrainingData, my_model: ModelBase, encoder: LabelEncoder
) -> tuple[TrainingData, ModelBase]:
    mlflow.set_tracking_uri("http://localhost:5000")

    logger.info("Trying to establish connection to MLFlow server...")
    mlflow.set_experiment("domain-classification")
    logger.info("Connection established")

    mlflow.sklearn.autolog()  # type: ignore

    with mlflow.start_run() as run:
        train_data, my_model = train_core(train_data, my_model)

        save_model_encoder_run_id(
            my_model.search.best_estimator_,
            encoder,
            run.info.run_id,
        )

        mlflow.set_tags({"model": my_model.model_name, "search": my_model.search_name})

    return train_data, my_model


def train_from_config(
    model_df: pd.DataFrame,
) -> tuple[LabelEncoder, TrainingData, ModelBase]:
    encoder, train_data = get_encoder_train_data(model_df)
    my_model = get_model_from_config()

    train_data, my_model = train_w_tracking(train_data, my_model, encoder)

    return encoder, train_data, my_model
