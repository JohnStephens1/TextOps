import logging

from text_classifier.config.logging_config import setup_logging

setup_logging()


logger = logging.getLogger("text_classifier.scripts.evaluate")


def main() -> None:
    logger.info("Evaluating...")

    # load model, encoder, run_id
    # get metrics, figs
    # log metrics, figs, via mlflow run, also via dvc plots | metrics
    # potentially battle with contendors
    # out eval results (to feed into model register)

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
