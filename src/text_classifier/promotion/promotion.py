import logging
from typing import Any

import mlflow

from text_classifier.protocols import Predictor

logger = logging.getLogger(__name__)


def get_champ_metrics() -> dict[str, float]:
    client = mlflow.MlflowClient("http://localhost:5000")

    try:
        champion_version = client.get_model_version_by_alias(
            "text_classifier", "champion"
        )
    except mlflow.exceptions.MlflowException:  # type: ignore
        logger.warning("Champion model not found. Defaulting to empty dict.")
        return {}

    if champion_version.run_id is None:
        logger.warning("Champion run id not found. Defaulting to empty dict.")
        return {}

    run = client.get_run(champion_version.run_id)
    metrics = run.data.metrics

    return metrics


def check_class_recall_eligibility(
    cfg: dict[str, Any], test_metrics: dict[str, Any]
) -> bool:
    for name, value in test_metrics.items():
        if (
            name.startswith("test_recall_class_")
            and value < cfg["recall_per_class"]["min"]
        ):
            logger.info(
                f"Promotion not eligible. Class recall insufficient for class: {name}"
                f"Result: {value:.4f} < {cfg['recall_per_class']['min']}"
            )
            return False

    return True


def check_f1_eligibility(cfg: dict[str, Any], test_metrics: dict[str, Any]) -> bool:
    champ_metrics = get_champ_metrics()

    if (
        test_metrics["test_f1"] - champ_metrics.get("test_f1", 0)
        < cfg["f1_macro"]["min_improvement"]
    ):
        logger.info(
            "Promotion not eligible. F1 improvement insufficient.\n"
            f"Contendor: {test_metrics['test_f1']}\n"
            f"Champion: {champ_metrics.get('test_f1')}\n"
            f"Difference: {test_metrics['test_f1'] - champ_metrics.get('test_f1', 0)} < {cfg['f1_macro']['min_improvement']}"
        )
        return False

    return True


def check_promotion_eligibility(
    cfg: dict[str, Any], test_metrics: dict[str, Any]
) -> bool:
    return check_class_recall_eligibility(cfg, test_metrics) and check_f1_eligibility(
        cfg, test_metrics
    )


def promote(model: Predictor) -> None:
    # TODO implement promotion
    # history of past contendors?
    pass
