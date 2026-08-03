import tempfile
from pathlib import Path

import joblib  # type: ignore
import mlflow
import pandas as pd
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder

from text_classifier.data.model import get_encoder_train_data
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.evaluation.plots import get_model_eval_figs
from text_classifier.model.models import (
    ModelBase,
    get_model_from_config,
)
from text_classifier.model.predictions import get_predictions_w_encoder
from text_classifier.schema import TrainingData


def log_encoder(
    encoder: LabelEncoder,
    encoder_file_name: str = "label_encoder.joblib",
    artifact_dir_name: str = "preprocessing",
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder_path = Path(tmp_dir) / encoder_file_name

        joblib.dump(encoder, encoder_path)
        mlflow.log_artifact(str(encoder_path), artifact_path=artifact_dir_name)


def log_metrics_figs(metrics: dict[str, float], figs: dict[str, Figure]):
    mlflow.log_metrics(metrics)

    for fig_name, fig in figs.items():
        mlflow.log_figure(fig, f"figures/{fig_name}.png")


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
) -> tuple[TrainingData, ModelBase, dict[str, float], dict[str, Figure]]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("domain-classification")
    mlflow.sklearn.autolog()  # type: ignore

    with mlflow.start_run():
        train_data, my_model = train_core(train_data, my_model)

        metrics, figs = get_metrics_figs(my_model, train_data, encoder)
        log_metrics_figs(metrics, figs)
        log_encoder(encoder)

        mlflow.set_tags({"model": my_model.model_name, "search": my_model.search_name})

    return train_data, my_model, metrics, figs


def train_from_config(
    feature_df: pd.DataFrame,
) -> tuple[LabelEncoder, TrainingData, ModelBase, dict[str, float], dict[str, Figure]]:
    encoder, train_data = get_encoder_train_data(feature_df)
    my_model = get_model_from_config()

    train_data, my_model, metrics, figs = train_w_tracking(
        train_data, my_model, encoder
    )

    return encoder, train_data, my_model, metrics, figs
