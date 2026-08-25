import logging

import mlflow
import pandas as pd
from common.env_loader import get_env_var
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    EMBEDDING_MODEL_STR_ARTIFACT_PATH,
    ENCODER_ARTIFACT_PATH,
    FIGS_DIR,
    PLOTS_DIR,
    TEST_METRICS_PATH,
    TRAIN_BEST_ESTIMATOR_PATH,
    TRAIN_RUN_ID_PATH,
)
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.evaluation.plots import get_model_eval_figs, get_model_eval_plots
from text_classifier.model.predictions import get_predictions_w_encoder
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_joblib, load_text, save
from text_classifier.schema import XYData

logger = logging.getLogger(__name__)


MLFLOW_URL = get_env_var("MLFLOW_URL")


def save_metrics_plots_figs(
    metrics: dict[str, float],
    plots: dict[str, pd.DataFrame],
    figs: dict[str, Figure],
) -> None:
    save(metrics, TEST_METRICS_PATH)

    for plot_name, df in plots.items():
        save(df, PLOTS_DIR / f"{plot_name}.csv")

    for fig_name, fig in figs.items():
        save(fig, FIGS_DIR / f"{fig_name}.png")


def load_model_encoder_run_id() -> tuple[Predictor, LabelEncoder, str]:
    model = load_joblib(TRAIN_BEST_ESTIMATOR_PATH)
    label_encoder = load_joblib(ENCODER_ARTIFACT_PATH)
    run_id = load_text(TRAIN_RUN_ID_PATH)

    return model, label_encoder, run_id


def log_encoder(artifact_dir_name: str = "preprocessing") -> None:
    mlflow.log_artifact(str(ENCODER_ARTIFACT_PATH), artifact_path=artifact_dir_name)


def log_embedding_model_str(artifact_dir_name: str = "embeddings") -> None:
    mlflow.log_artifact(
        str(EMBEDDING_MODEL_STR_ARTIFACT_PATH), artifact_path=artifact_dir_name
    )


def log_metrics_figs(metrics: dict[str, float], figs: dict[str, Figure]) -> None:
    mlflow.log_metrics(metrics)

    for fig_name, fig in figs.items():
        mlflow.log_figure(fig, f"figures/{fig_name}.png")


def get_metrics_plots_figs(
    model: Predictor, ds: XYData, encoder: LabelEncoder
) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Figure]]:
    predictions_w_encoder = get_predictions_w_encoder(model, ds, encoder)

    metrics = get_classification_metrics(
        predictions_w_encoder.predictions,
        prefix="test",
    )

    plots = get_model_eval_plots(predictions_w_encoder)
    figs = get_model_eval_figs(predictions_w_encoder)

    return metrics, plots, figs


def evaluate(
    model: Predictor, ds: XYData, encoder: LabelEncoder, run_id: str
) -> tuple[dict[str, float], dict[str, pd.DataFrame], dict[str, Figure]]:
    mlflow.set_tracking_uri(MLFLOW_URL)

    logger.info("Trying to establish connection to MLFlow server...")
    with mlflow.start_run(run_id):
        logger.info("Connection established")

        metrics, plots, figs = get_metrics_plots_figs(model, ds, encoder)

        log_metrics_figs(metrics, figs)
        log_encoder()
        log_embedding_model_str()

    return metrics, plots, figs
