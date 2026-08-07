from collections.abc import Callable
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


def _binarize_y_true(
    preds_w_encoder: PredictionsEncoder,
) -> np.typing.NDArray[np.float64]:
    n_classes = preds_w_encoder.encoder.classes_.shape[0]

    y_true_bin = np.asarray(
        label_binarize(preds_w_encoder.predictions.y_true, classes=range(n_classes)),
        dtype=np.float64,
    )

    return y_true_bin


def _multiclass_fig_body(
    display_cls: type[PrecisionRecallDisplay | RocCurveDisplay],
    preds_w_encoder: PredictionsEncoder,
    title: str,
    x_label: str,
    y_label: str,
    with_random_line: bool = False,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    y_true_bin = _binarize_y_true(preds_w_encoder)

    for i, cls in enumerate(preds_w_encoder.encoder.classes_):
        display_cls.from_predictions(
            y_true_bin[:, i],
            preds_w_encoder.predictions.y_proba[:, i],
            name=cls,
            ax=ax,
        )

    if with_random_line:
        ax.plot([0, 1], [0, 1], "k--", label="random")

    ax.set_title(title, pad=16)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend(loc="lower right")
    ax.grid()

    return fig


def get_roc_curve_fig(
    preds_w_encoder: PredictionsEncoder,
) -> Figure:
    return _multiclass_fig_body(
        display_cls=RocCurveDisplay,
        preds_w_encoder=preds_w_encoder,
        title="Multiclass ROC Curve",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        with_random_line=True,
    )


def get_precision_recall_curve_fig(
    preds_w_encoder: PredictionsEncoder,
) -> Figure:
    return _multiclass_fig_body(
        display_cls=PrecisionRecallDisplay,
        preds_w_encoder=preds_w_encoder,
        title="Multiclass Precision-Recall Curve",
        x_label="Recall",
        y_label="Precision",
        with_random_line=False,
    )


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


def _multiclass_plot_body(
    curve_fn: Callable[
        [np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
        tuple[
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
        ],
    ],
    preds_w_encoder: PredictionsEncoder,
    col_name_1: str,
    col_name_2: str,
) -> pd.DataFrame:
    """curve_fn supports roc_curve, precision_recall_curve"""

    rows = []

    y_true_bin = _binarize_y_true(preds_w_encoder)

    for i, cls in enumerate(preds_w_encoder.encoder.classes_):
        x, y, _ = curve_fn(
            y_true_bin[:, i],
            preds_w_encoder.predictions.y_proba[:, i],
        )

        for s, t in zip(x, y):
            rows.append({"class": cls, col_name_1: s, col_name_2: t})

    df = pd.DataFrame(rows)

    return df


def get_precision_recall_curve_plot_df(
    preds_w_encoder: PredictionsEncoder,
) -> pd.DataFrame:
    return _multiclass_plot_body(
        precision_recall_curve, preds_w_encoder, "precision", "recall"
    )


def get_roc_curve_plot_df(preds_w_encoder: PredictionsEncoder) -> pd.DataFrame:
    return _multiclass_plot_body(roc_curve, preds_w_encoder, "fpr", "tpr")


def get_precision_recall_curve_plot_fig(df: pd.DataFrame) -> Figure:
    return _group_plot(
        df,
        "recall",
        "precision",
        "Multiclass Precision-Recall Curve",
        "Recall",
        "Precision",
    )


def get_roc_curve_plot_fig(df: pd.DataFrame) -> Figure:
    return _group_plot(
        df,
        "fpr",
        "tpr",
        "Multiclass ROC Curve",
        "False Positive Rate",
        "True Positive Rate",
        True,
    )


def _group_plot(
    df: pd.DataFrame,
    col_1_name: str,
    col_2_name: str,
    title: str,
    x_label: str,
    y_label: str,
    with_random_line: bool = False,
    class_col: str = "class",
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    for cls, group in df.groupby(class_col):
        ax.plot(group[col_1_name], group[col_2_name], label=f"{cls}")

    if with_random_line:
        ax.plot([0, 1], [0, 1], "k--", label="random")

    ax.set_title(title, pad=16)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend(loc="lower right")
    ax.grid()

    return fig


def get_model_eval_plots(
    preds_w_encoder: PredictionsEncoder,
) -> dict[str, Figure]:
    return get_roc_curve_plot_df(preds_w_encoder)
