import logging

from text_classifier.config.config import (
    X_TEST_PATH,
    Y_TEST_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.loader import get_x_y_data
from text_classifier.evaluation.eval import (
    evaluate,
    load_model_encoder_run_id,
    save_metrics_plots_figs,
)

setup_logging()


logger = logging.getLogger("text_classifier.scripts.evaluate")


def main() -> None:
    """evaluates and tracks model experiment

    In detail:
    - loads the model, label_encoder, run_id, test dataset
    - creates metrics, plots, figs from trained model
    - tracks model, metrics, plots, figs using MLFlow and DVC
    """

    logger.info("Evaluating...")

    model, label_encoder, run_id = load_model_encoder_run_id()
    ds = get_x_y_data(X_TEST_PATH, Y_TEST_PATH)

    metrics, plots, figs = evaluate(model, ds, label_encoder, run_id)

    save_metrics_plots_figs(
        metrics,
        plots,
        figs,
    )

    logger.info("Saved metrics, plots, figs")

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
