import logging

from text_classifier.config.logging_config import setup_logging
from text_classifier.model.save_load import load_model_encoder_run_id

setup_logging()


logger = logging.getLogger("text_classifier.scripts.evaluate")


def main() -> None:
    """trains, evaluates and tracks model experiment end-to-end.

    In detail:
    - loads the model, label_encoder, run_id
    - creates metrics, plots from trained model
    - evaluates model based on performance
    - tracks model, metrics, plots using MLFlow and DVC
    """

    logger.info("Evaluating...")

    model, encoder, run_id = load_model_encoder_run_id()

    # get metrics, figs
    # log metrics, figs, via mlflow run, also via dvc plots | metrics
    # potentially battle with contendors
    # out eval results (to feed into model register)

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
