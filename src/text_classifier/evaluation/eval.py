import tempfile
from pathlib import Path

import joblib  # type: ignore
import mlflow
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder

from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.evaluation.plots import get_model_eval_figs
from text_classifier.model.predictions import get_predictions_w_encoder
from text_classifier.protocols import Predictor
from text_classifier.schema import TrainingData


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


def get_metrics_figs(
    model: Predictor, train_data: TrainingData, encoder: LabelEncoder
) -> tuple[dict[str, float], dict[str, Figure]]:
    predictions_w_encoder = get_predictions_w_encoder(
        model, train_data.X_test, train_data.y_test, encoder
    )

    metrics = get_classification_metrics(
        predictions_w_encoder.predictions,
        prefix="test",
    )

    figs = get_model_eval_figs(predictions_w_encoder)

    return metrics, figs


def evaluate(
    train_data: TrainingData, model: Predictor, encoder: LabelEncoder, run_id: str
) -> tuple[TrainingData, Predictor, dict[str, float], dict[str, Figure]]:
    mlflow.set_tracking_uri("http://localhost:5000")

    print("trying 2 connect")  # TODO replace with logger

    with mlflow.start_run(run_id):
        metrics, figs = get_metrics_figs(model, train_data, encoder)

        log_metrics_figs(metrics, figs)
        log_encoder(encoder)

    return train_data, model, metrics, figs
