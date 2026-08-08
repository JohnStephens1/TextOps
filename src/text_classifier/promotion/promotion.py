import logging
from typing import Any

from text_classifier.protocols import Predictor

logger = logging.getLogger(__name__)


def check_promotion_eligibility(
    cfg: dict[str, Any], test_metrics: dict[str, Any]
) -> bool:
    for name, value in test_metrics.items():
        if (
            name.startswith("test_recall_class_")
            and value < cfg["class_recall"]["minimum"]
        ):
            logger.info(
                f"Promotion not eligible: {name} < {cfg['class_recall']['minimum']} : {value:.4f}"
            )
            return False

    # current best required
    # TODO check for default
    # new - current > min_improvement
    return test_metrics["test_f1"] > cfg["macro_f1"]["min_improvement"]


def promote(model: Predictor):
    pass
