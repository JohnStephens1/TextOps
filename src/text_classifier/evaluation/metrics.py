import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from text_classifier.schema import Predictions


def get_classification_metrics(
    preds: Predictions,
    prefix: str = "",
) -> dict[str, float]:

    # TODO add actual class label via encoder
    recall_per_class = np.asarray(
        recall_score(
            preds.y_true,
            preds.y_pred,
            average=None,
        )
    )

    metrics = {
        "accuracy": accuracy_score(preds.y_true, preds.y_pred),
        "f1": f1_score(preds.y_true, preds.y_pred, average="macro"),
        "precision": precision_score(preds.y_true, preds.y_pred, average="macro"),
        "recall": recall_score(preds.y_true, preds.y_pred, average="macro"),
        "roc_auc": roc_auc_score(
            preds.y_true, preds.y_proba, multi_class="ovo", average="macro"
        ),
        **{
            f"recall_class_{i}": float(recall)
            for i, recall in enumerate(recall_per_class)
        },
    }

    if prefix != "":
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    return metrics


def print_pred_report(
    preds: Predictions,
    prefix: str = "",
) -> None:
    metrics = get_classification_metrics(preds, prefix)

    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1: {metrics['f1']}")
    print(f"ROC AUC: {metrics['roc_auc']}")

    print()

    print(
        f"Classification Report:\n{classification_report(preds.y_true, preds.y_pred)}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(preds.y_true, preds.y_pred)}")
