import logging

import joblib  # type: ignore
import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    ENCODER_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    RUN_ID_ARTIFACT_PATH,
)
from text_classifier.data.train_data import get_encoder_train_data
from text_classifier.model.models import (
    ModelBase,
    get_model_from_config,
)
from text_classifier.schema import TrainingData

logger = logging.getLogger(__name__)


def save_model_encoder_run_id(model: BaseEstimator, encoder: LabelEncoder, run_id: str):
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    joblib.dump(encoder, ENCODER_ARTIFACT_PATH)

    RUN_ID_ARTIFACT_PATH.write_text(run_id, encoding="utf-8")


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
