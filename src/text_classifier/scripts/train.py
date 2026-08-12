import logging

from text_classifier.config.config import (
    TRAIN_MODEL_VERSION_PATH,
    TRAIN_RUN_ID_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.loader import get_x_y_data
from text_classifier.model.training import save_search_results, train_from_config
from text_classifier.save_load import save

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

    ds = get_x_y_data(X_TRAIN_PATH, Y_TRAIN_PATH)

    my_model, run_id, model_version = train_from_config(ds)

    save_search_results(my_model)
    logger.info("Saved model")

    save(model_version, TRAIN_MODEL_VERSION_PATH)
    logger.info("Saved model version")

    save(run_id, TRAIN_RUN_ID_PATH)
    logger.info("Saved run id")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
