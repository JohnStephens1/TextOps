import logging
from typing import Any

from text_classifier.config.config import (
    PRODUCTION_MODEL_METRICS_PATH,
    PRODUCTION_MODEL_PATH,
)
from text_classifier.protocols import Predictor
from text_classifier.save_load import load_json

logger = logging.getLogger(__name__)


def check_class_recall_eligibility(
    cfg: dict[str, Any], test_metrics: dict[str, Any]
) -> bool:
    for name, value in test_metrics.items():
        if (
            name.startswith("test_recall_class_")
            and value < cfg["class_recall"]["minimum"]
        ):
            logger.info(
                f"Promotion not eligible. Class recall insufficient for class: {name}"
                f"Result: {value:.4f} < {cfg['class_recall']['minimum']}"
            )
            return False

    return True


def check_promotion_eligibility(
    cfg: dict[str, Any], test_metrics: dict[str, Any]
) -> bool:
    if not check_class_recall_eligibility(cfg, test_metrics):
        return False

    # TODO extract, check, compare
    # handle "no production model"
    # handle "key not found"
    if PRODUCTION_MODEL_PATH.exists() and PRODUCTION_MODEL_METRICS_PATH.exists():
        production_metrics = load_json(PRODUCTION_MODEL_METRICS_PATH)

        if (
            test_metrics["test_f1"] - production_metrics["test_f1"]
            < cfg["macro_f1"]["min_improvement"]
        ):
            logger.info(
                "Promotion not eligible. F1 improvement insufficient.\n"
                f"Contendor: {test_metrics['test_f1']}\n"
                f"Production: {production_metrics['test_f1']}\n"
                f"Difference: {test_metrics['test_f1'] - production_metrics['test_f1']} < {cfg['macro_f1']['min_improvement']}"
            )
            return False

    return True


def promote(model: Predictor) -> None:
    # TODO implement promotion
    # history of past contendors?
    pass
