from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from text_classifier.data.model import get_encoder_train_data
from text_classifier.evaluation.metrics import get_classification_metrics
from text_classifier.model.models import (
    ModelBase,
)
from text_classifier.model.pipeline import get_model_pipe
from text_classifier.schema import TrainingData


def get_cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def get_cv_score(
    pipe: Pipeline, X: pd.DataFrame, y: np.typing.ArrayLike
) -> np.typing.ArrayLike:
    return cross_val_score(
        pipe,
        X,
        y,
        cv=get_cv_splitter(),
        scoring="f1_macro",
        verbose=1,
    )


def get_random_search(
    pipe: Pipeline,
    param_distribution: dict[str, Any],
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        pipe,
        param_distribution,
        n_iter=10,
        cv=get_cv_splitter(),
        scoring="f1_macro",
        random_state=42,
        verbose=1,
    )


# TODO associate type of search with model
# GridSearchCV
# RandomizedSearchCV


# Next up:
# isolate train_core
# wrap for mlflow tracking

# TODO
# dump encoder with joblib
# add to mlflow with
#   mlflow.log_artifact(encoder_path, artifact_path="preprocessors")

# add prefix to metrics, this case "test_..."
# precision, recall
# mb confusion matrix, plots


def train_core(
    train_data: TrainingData,
    my_model: ModelBase,
) -> tuple[TrainingData, RandomizedSearchCV, dict[str, Any]]:
    pipe = get_model_pipe(my_model.model)

    search = get_random_search(pipe, my_model.default_param_dist_w_model_prefix)
    search.fit(train_data.X_train, train_data.y_train)

    metrics = get_classification_metrics(
        train_data.y_test,
        search.predict(train_data.X_test),
        search.predict_proba(train_data.X_test),
    )

    return train_data, search, metrics


def train_w_tracking(
    train_data: TrainingData,
    my_model: ModelBase,
) -> tuple[TrainingData, RandomizedSearchCV, dict[str, Any]]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.sklearn.autolog()  # type: ignore

    mlflow.set_experiment("domain-classification")
    # mlflow.set_tags({
    #     "model": "RandomForestClassifier",
    #     "search": "GridSearchCV"
    # })

    with mlflow.start_run():
        train_data, search, metrics = train_core(train_data, my_model)

        mlflow.log_metrics(metrics)
        # mlflow.log_metric("cv_score", search.best_score_)
        # mlflow.sklearn.log_model(search.best_estimator_, "model")
        # mlflow.sklearn.log_model(search, "search")

    return train_data, search, metrics


def train_qm(
    my_model: ModelBase,
) -> tuple[LabelEncoder, TrainingData, RandomizedSearchCV, dict[str, Any]]:
    encoder, train_data = get_encoder_train_data()
    train_data, search, metrics = train_w_tracking(train_data, my_model)

    return encoder, train_data, search, metrics
