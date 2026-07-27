from typing import Any

import mlflow
from sklearn.preprocessing import LabelEncoder

from text_classifier.data.model import get_encoder_train_data
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.model.models import (
    ModelBase,
)
from text_classifier.schema import TrainingData

# TODO
# dump encoder with joblib
# add to mlflow with
#   mlflow.log_artifact(encoder_path, artifact_path="preprocessors")

# for metrics, potentially add
# confusion matrix
# roc curve
# precision recall curve
# feature importance


def train_core(
    train_data: TrainingData,
    my_model: ModelBase,
) -> tuple[TrainingData, ModelBase, dict[str, Any]]:
    my_model.search.fit(train_data.X_train, train_data.y_train)

    metrics = get_classification_metrics(
        train_data.y_test,
        my_model.search.predict(train_data.X_test),
        my_model.search.predict_proba(train_data.X_test),
        prefix="test",
    )

    return train_data, my_model, metrics


def train_w_tracking(
    train_data: TrainingData,
    my_model: ModelBase,
) -> tuple[TrainingData, ModelBase, dict[str, Any]]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("domain-classification")
    mlflow.sklearn.autolog()  # type: ignore

    with mlflow.start_run():
        train_data, my_model, metrics = train_core(train_data, my_model)

        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model": my_model.model_name, "search": my_model.search_name})

    return train_data, my_model, metrics


def train_qm(
    my_model: ModelBase,
) -> tuple[LabelEncoder, TrainingData, ModelBase, dict[str, Any]]:
    encoder, train_data = get_encoder_train_data()
    train_data, my_model, metrics = train_w_tracking(train_data, my_model)

    return encoder, train_data, my_model, metrics
