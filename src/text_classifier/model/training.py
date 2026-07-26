import typing

import mlflow  # type: ignore
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
    param_distribution: dict[str, typing.Any],
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


def train_qm(
    my_model: ModelBase,
) -> tuple[LabelEncoder, TrainingData, RandomizedSearchCV]:
    encoder, train_data = get_encoder_train_data()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        print(run.info.artifact_uri)
        pipe = get_model_pipe(my_model.model)

        search = get_random_search(pipe, my_model.default_param_dist_w_model_prefix)
        search.fit(train_data.X_train, train_data.y_train)

        metrics = get_classification_metrics(
            train_data.y_test,
            search.predict(train_data.X_test),
            search.predict_proba(train_data.X_test),
        )

        mlflow.log_metrics(metrics)
        # mlflow.log_metric("cv_score", search.best_score_)
        # mlflow.sklearn.log_model(search.best_estimator_, "model")
        # mlflow.sklearn.log_model(search, "search")

    return encoder, train_data, search
