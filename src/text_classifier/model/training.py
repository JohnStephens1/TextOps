import logging

import mlflow
import pandas as pd

from text_classifier.config.config import (
    TRAIN_BEST_ESTIMATOR_PATH,
    TRAIN_BEST_PARAMS_PATH,
    TRAIN_BEST_SCORE_PATH,
    TRAIN_CV_RESULTS_PATH,
)
from text_classifier.model.models import (
    ModelBase,
    get_model_from_config,
)
from text_classifier.save_load import save
from text_classifier.schema import XYData

logger = logging.getLogger(__name__)


def save_search_results(my_model: ModelBase) -> None:
    save(my_model.search.best_estimator_, TRAIN_BEST_ESTIMATOR_PATH)
    save(my_model.search.best_params_, TRAIN_BEST_PARAMS_PATH)
    save(str(my_model.search.best_score_), TRAIN_BEST_SCORE_PATH)

    cv_results_df = (
        pd.DataFrame(my_model.search.cv_results_)
        .sort_values("rank_test_score")
        .convert_dtypes()
    )

    save(cv_results_df, TRAIN_CV_RESULTS_PATH, index=False)


def train_core(
    train_ds: XYData,
    my_model: ModelBase,
) -> ModelBase:
    my_model.search.fit(train_ds.X, train_ds.y)

    return my_model


def train_w_tracking(train_ds: XYData, my_model: ModelBase) -> tuple[ModelBase, str]:
    mlflow.set_tracking_uri("http://localhost:5000")

    logger.info("Trying to establish connection to MLFlow server...")
    mlflow.set_experiment("domain-classification")
    logger.info("Connection established")

    mlflow.sklearn.autolog()  # type: ignore

    with mlflow.start_run() as run:
        my_model = train_core(train_ds, my_model)

        run_id: str = run.info.run_id

        mlflow.set_tags({"model": my_model.model_name, "search": my_model.search_name})

    return my_model, run_id


def train_from_config(train_ds: XYData) -> tuple[ModelBase, str]:
    my_model = get_model_from_config()
    my_model, run_id = train_w_tracking(train_ds, my_model)

    return my_model, run_id
