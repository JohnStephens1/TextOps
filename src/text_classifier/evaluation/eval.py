import logging
import tempfile
from pathlib import Path

import joblib  # type: ignore
import mlflow
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    ENCODER_ARTIFACT_PATH,
    TRAIN_BEST_ESTIMATOR_PATH,
    TRAIN_RUN_ID,
)
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.evaluation.plots import get_model_eval_figs
from text_classifier.model.predictions import get_predictions_w_encoder
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_joblib, load_text
from text_classifier.schema import XYData

logger = logging.getLogger(__name__)


def load_model_encoder_run_id() -> tuple[Predictor, LabelEncoder, str]:
    model = load_joblib(TRAIN_BEST_ESTIMATOR_PATH)
    label_encoder = load_joblib(ENCODER_ARTIFACT_PATH)
    run_id = load_text(TRAIN_RUN_ID)

    return model, label_encoder, run_id


# TODO review
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
    model: Predictor, ds: XYData, encoder: LabelEncoder
) -> tuple[dict[str, float], dict[str, Figure]]:
    predictions_w_encoder = get_predictions_w_encoder(model, ds, encoder)

    metrics = get_classification_metrics(
        predictions_w_encoder.predictions,
        prefix="test",
    )

    figs = get_model_eval_figs(predictions_w_encoder)

    return metrics, figs


def evaluate(
    ds: XYData, model: Predictor, encoder: LabelEncoder, run_id: str
) -> tuple[Predictor, dict[str, float], dict[str, Figure]]:
    mlflow.set_tracking_uri("http://localhost:5000")

    logger.info("Trying to establish connection to MLFlow server...")
    with mlflow.start_run(run_id):
        logger.info("Connection established")

        metrics, figs = get_metrics_figs(model, ds, encoder)

        log_metrics_figs(metrics, figs)
        log_encoder(encoder)

    return model, metrics, figs
