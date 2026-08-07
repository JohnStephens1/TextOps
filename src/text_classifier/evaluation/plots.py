from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from text_classifier.model.models import ModelBase
from text_classifier.schema import Predictions, PredictionsEncoder, XYData


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
def print_perm_importance(
    my_model: ModelBase,
    test_ds: XYData,
    n_repeats: int = 10,
    random_state: int = 42,
) -> None:
    print(f"scorer: {my_model.search.scorer_}")

    result = permutation_importance(
        my_model.search,
        test_ds.X,
        test_ds.y,
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


def get_confusion_matrix_plot(preds_w_encoder: PredictionsEncoder) -> pd.DataFrame:
    cm = confusion_matrix(
        preds_w_encoder.predictions.y_true, preds_w_encoder.predictions.y_pred
    )

    index = [f"true_{i}" for i in range(len(preds_w_encoder.encoder.classes_))]
    cols = [f"pred_{i}" for i in range(len(preds_w_encoder.encoder.classes_))]

    return pd.DataFrame(
        cm,
        index=index,
        columns=cols,
    )


# pyright: basic
def get_precision_recall_curve_plot(preds_w_encoder: PredictionsEncoder):
    prc_rows = []

    for i, cls in enumerate(preds_w_encoder.encoder.classes_):
        precision, recall, _ = precision_recall_curve(
            preds_w_encoder.predictions.y_true[:, i],
            preds_w_encoder.predictions.y_proba[:, i],
        )

        for p, r in zip(precision, recall):
            prc_rows.append({"class": cls, "precision": p, "recall": r})

    prc_df = pd.DataFrame(prc_rows)

    return prc_df


def get_roc_curve_plot(preds_w_encoder: PredictionsEncoder):
    roc_rows = []

    n_classes = preds_w_encoder.encoder.classes_.shape[0]

    y_test_bin = np.asarray(
        label_binarize(preds_w_encoder.predictions.y_true, classes=range(n_classes)),
        dtype=np.float64,
    )

    for i, cls in enumerate(preds_w_encoder.encoder.classes_):
        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            preds_w_encoder.predictions.y_proba[:, i],
        )

        for f, t in zip(fpr, tpr):
            roc_rows.append({"class": cls, "fpr": f, "tpr": t})

    roc_df = pd.DataFrame(roc_rows)

    return roc_df


def group_plot(df: pd.DataFrame):
    plt.figure(figsize=(7, 7))

    for cls, group in df.groupby("class"):
        plt.plot(group["fpr"], group["tpr"], label=f"{cls}")

    plt.plot([0, 1], [0, 1], "k--", label="random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()


def get_model_eval_plots(
    preds_w_encoder: PredictionsEncoder,
) -> dict[str, Figure]:
    return get_roc_curve_plot(preds_w_encoder)
