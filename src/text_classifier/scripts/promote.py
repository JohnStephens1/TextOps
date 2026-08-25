import logging

from common.logging_config import setup_logging

from text_classifier.config.config import (
    CONFIG_PATH,
    TEST_METRICS_PATH,
    TRAIN_MODEL_VERSION_PATH,
)
from text_classifier.promotion.promotion import check_promotion_eligibility, promote
from text_classifier.save_load import load_json, load_text, load_yaml

setup_logging()


logger = logging.getLogger("text_classifier.scripts.promote")


def main() -> None:
    logger.info("Checking promotion eligibility...")

    cfg = load_yaml(CONFIG_PATH / "promote.yaml")
    train_model_version = load_text(TRAIN_MODEL_VERSION_PATH)

    # load train metrics
    test_metrics = load_json(TEST_METRICS_PATH)

    # TODO add more train metrics
    # TODO update test metrics to relevant ones

    is_eligible = check_promotion_eligibility(cfg, test_metrics)

    if is_eligible:
        promote(train_model_version)
        logger.info(f"Model {train_model_version} has been promoted!")
    else:
        logger.info("Model performance insufficient for promotion")


if __name__ == "__main__":
    main()
