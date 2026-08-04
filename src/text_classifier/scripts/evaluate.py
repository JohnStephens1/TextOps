import logging

from text_classifier.config.logging_config import setup_logging

# from text_classifier.model.save_load import load_model_encoder_run_id
# from text_classifier.evaluation.eval import evaluate

setup_logging()


logger = logging.getLogger("text_classifier.scripts.evaluate")


def main() -> None:
    """evaluates and tracks model experiment

    In detail:
    - loads the model, label_encoder, run_id
    - creates metrics, plots from trained model
    - evaluates model based on performance
    - tracks model, metrics, plots using MLFlow and DVC
    """

    logger.info("Evaluating...")

    # model, encoder, run_id = load_model_encoder_run_id()

    # i need data, at least test data...
    # TODO new data split step it is
    # evaluate()

    # get metrics, figs
    # log metrics, figs, via mlflow run, also via dvc plots | metrics
    # potentially battle with contendors
    # out eval results (to feed into model register)

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
