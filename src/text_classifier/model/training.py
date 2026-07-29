import mlflow
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder

from text_classifier.data.model import get_encoder_train_data
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.evaluation.plots import get_model_eval_figs
from text_classifier.model.models import (
    ModelBase,
    get_predictions_w_encoder,
)
from text_classifier.schema import TrainingData

# TODO
# dump encoder with joblib
# add to mlflow with
#   mlflow.log_artifact(encoder_path, artifact_path="preprocessors")


def log_metrics_figs(metrics: dict[str, float], figs: dict[str, Figure]):
    mlflow.log_metrics(metrics)

    for fig_name, fig in figs.items():
        mlflow.log_figure(fig, f"figures/{fig_name}.png")


def get_metrics_figs(
    my_model: ModelBase, train_data: TrainingData, encoder: LabelEncoder
) -> tuple[dict[str, float], dict[str, Figure]]:
    predictions_w_encoder = get_predictions_w_encoder(
        my_model.search, train_data.X_test.to_numpy(), train_data.y_test, encoder
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

        mlflow.set_tags({"model": my_model.model_name, "search": my_model.search_name})

    return train_data, my_model, metrics, figs


def train_qm(
    my_model: ModelBase,
) -> tuple[LabelEncoder, TrainingData, ModelBase, dict[str, float], dict[str, Figure]]:
    encoder, train_data = get_encoder_train_data()
    train_data, my_model, metrics, figs = train_w_tracking(
        train_data, my_model, encoder
    )

    return encoder, train_data, my_model, metrics, figs
