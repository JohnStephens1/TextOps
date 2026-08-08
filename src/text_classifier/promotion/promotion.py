from typing import Any

from text_classifier.protocols import Predictor


def check_promotion_eligibility(
    cfg: dict[str, Any], test_metrics: dict[str, Any]
) -> bool:
    for metric in test_metrics["class_recall"]:
        if metric < cfg["class_recall"]["minimum"]:
            return False

    # current best required
    # TODO check for default
    # new - current > min_improvement
    return test_metrics["macro_f1"] > cfg["macro_f1"]["min_improvement"]


def promote(model: Predictor):
    pass
