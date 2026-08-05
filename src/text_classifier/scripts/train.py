import logging

from text_classifier.config.config import (
    TRAIN_RUN_ID,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.model.training import save_search_results, train_from_config
from text_classifier.save_load import load_parquet, save
from text_classifier.schema import XYData

setup_logging()


logger = logging.getLogger("text_classifier.scripts.train")


def main() -> None:
    """trains the model based on the configuration in params.yaml, tracked via mlflow, and saves the search results and run_id.

    In detail:
    - loads the train splits
    - trains model from params.yaml config
    - tracks model using MLFlow
    - saves search results
        - best_estimator_
        - best_params_
        - best_score_
        - cv_results_
    - saves run_id
    """

    logger.info("Setting up training...")

    ds = XYData(
        load_parquet(X_TRAIN_PATH),
        load_parquet(Y_TRAIN_PATH).iloc[:, 0].to_numpy(),
    )

    my_model, run_id = train_from_config(ds)

    save_search_results(my_model)
    logger.info("Saved model")

    save(run_id, TRAIN_RUN_ID)
    logger.info("Saved run id")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
