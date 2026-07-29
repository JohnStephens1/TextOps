from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize

from text_classifier.model.models import ModelBase
from text_classifier.schema import Predictions, PredictionsEncoder, TrainingData


def get_confusion_matrix_fig(
    preds: Predictions,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(
        preds.y_true,
        preds.y_pred,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title("Confusion Matrix", pad=16)

    plt.tight_layout()

    return fig


def get_roc_curve_fig(
    preds_w_encoder: PredictionsEncoder,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))

    n_classes = preds_w_encoder.encoder.classes_.shape[0]
    y_test_bin = np.asarray(
        label_binarize(preds_w_encoder.predictions.y_true, classes=range(n_classes)),
        dtype=np.float64,
    )

    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            preds_w_encoder.predictions.y_proba[:, i],
            name=f"{preds_w_encoder.encoder.classes_[i]}",
            ax=ax,
        )

    ax.set_title("ROC Curve", pad=16)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    plt.tight_layout()

    return fig


def get_precision_recall_curve_fig(
    preds_w_encoder: PredictionsEncoder,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))

    n_classes = preds_w_encoder.encoder.classes_.shape[0]
    y_test_bin = np.asarray(
        label_binarize(preds_w_encoder.predictions.y_true, classes=range(n_classes)),
        dtype=np.float64,
    )

    for i in range(n_classes):
        PrecisionRecallDisplay.from_predictions(
            y_test_bin[:, i],
            preds_w_encoder.predictions.y_proba[:, i],
            name=f"{preds_w_encoder.encoder.classes_[i]}",
            ax=ax,
        )

    ax.set_title("Precision-Recall Curve", pad=16)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    plt.tight_layout()

    return fig


# this took 27 minutes to run
# per class importance
# perm importance
def print_perm_importance(
    my_model: ModelBase,
    data: TrainingData,
    n_repeats: int = 10,
    random_state: int = 42,
):
    print(f"scorer: {my_model.search.scorer_}")

    result = permutation_importance(
        my_model.search,
        data.X_test,
        data.y_test,
        scoring=my_model.search.scorer_,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    print(result)


def get_feature_importances_names(
    feature_importances: np.typing.NDArray[np.float64],
    feature_names: np.typing.NDArray[np.object_],
    merge_text_embeddings: bool = True,
    top_n: int = 5,
) -> tuple[Any, Any]:
    # feature_importances = my_model.search.best_estimator_.named_steps['model'].feature_importances_
    # feature_names = train_data.X_train.columns

    importances_dict = {x: y for x, y in zip(feature_names, feature_importances)}

    if merge_text_embeddings:
        text_keys = [k for k in importances_dict if k.startswith("text_")]

        if text_keys:
            importances_dict["text"] = max(
                np.float64(importances_dict[k]) for k in text_keys
            )

            for k in text_keys:
                del importances_dict[k]

    names, importances = zip(
        *sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # *sorted(importances_dict.items(), key=lambda x: x[1])[-top_n:]
    )

    return names, importances


def get_tree_based_feature_importance_fig(
    feature_importances: np.typing.NDArray[np.float64],
    feature_names: np.typing.NDArray[np.object_],
) -> Figure:
    """gets the feature importance figure for a tree-based model

    Args:
        feature_importances (np.typing.NDArray[np.float64]): my_model.search.best_estimator_.named_steps['model'].feature_importances_
        feature_names (np.typing.NDArray[np.object_]): train_data.X_train.columns.values

    Returns:
        Figure: tree-based feature importance figure
    """

    feat_names, feat_importances = get_feature_importances_names(
        feature_importances, feature_names
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.barh(feat_names, feat_importances, 0.5)

    ax.set_xscale("log")
    ax.set_title("Tree-Based Feature Importance", pad=16)
    ax.set_xlabel("Feature Importance (log scale)")

    plt.tight_layout()

    return fig


def get_model_eval_figs(preds_w_encoder: PredictionsEncoder) -> dict[str, Figure]:
    return {
        "confusion_matrix": get_confusion_matrix_fig(preds_w_encoder.predictions),
        "roc_curve": get_roc_curve_fig(preds_w_encoder),
        "precision_recall_curve": get_precision_recall_curve_fig(preds_w_encoder),
        # could add feature importance fig
    }
