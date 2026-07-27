from typing import Any

import mlflow
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
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


def get_grid_search(
    pipe: Pipeline,
    param_distribution: dict[str, Any],
) -> GridSearchCV:
    return GridSearchCV(
        pipe,
        param_distribution,
        cv=get_cv_splitter(),
        scoring="f1_macro",
        verbose=1,
    )


def get_search(
    pipe: Pipeline,
    param_dist: dict[str, Any],
    Search: type[RandomizedSearchCV | GridSearchCV],
    additional_params: dict[str, Any],
) -> RandomizedSearchCV | GridSearchCV:
    # for random search, add n_iter=10
    # could make current default arguments
    # even pass down to splitter
    return Search(
        pipe,
        param_dist,
        cv=get_cv_splitter(),
        scoring="f1_macro",
        verbose=1,
        **additional_params,
    )


# TODO associate type of search with model
# GridSearchCV
# RandomizedSearchCV
# properties
# search name
# model name


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
) -> tuple[TrainingData, RandomizedSearchCV, dict[str, Any]]:
    pipe = get_model_pipe(my_model.model)

    search = get_random_search(pipe, my_model.default_param_dist_w_model_prefix)
    search.fit(train_data.X_train, train_data.y_train)

    metrics = get_classification_metrics(
        train_data.y_test,
        search.predict(train_data.X_test),
        search.predict_proba(train_data.X_test),
        prefix="test",
    )

    return train_data, search, metrics


def train_w_tracking(
    train_data: TrainingData,
    my_model: ModelBase,
) -> tuple[TrainingData, RandomizedSearchCV, dict[str, Any]]:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("domain-classification")
    mlflow.sklearn.autolog()  # type: ignore

    with mlflow.start_run():
        train_data, search, metrics = train_core(train_data, my_model)

        mlflow.log_metrics(metrics)
        # mlflow.log_metric("cv_score", search.best_score_)
        # mlflow.sklearn.log_model(search.best_estimator_, "model")
        # mlflow.sklearn.log_model(search, "search")

        mlflow.set_tags(
            {
                "model": my_model.model_name,
                # "search": "GridSearchCV"
            }
        )

    return train_data, search, metrics


def train_qm(
    my_model: ModelBase,
) -> tuple[LabelEncoder, TrainingData, RandomizedSearchCV, dict[str, Any]]:
    encoder, train_data = get_encoder_train_data()
    train_data, search, metrics = train_w_tracking(train_data, my_model)

    return encoder, train_data, search, metrics
