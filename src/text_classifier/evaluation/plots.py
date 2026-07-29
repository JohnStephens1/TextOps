import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from text_classifier.model.models import ModelBase
from text_classifier.schema import TrainingData


def get_confusion_matrix_fig(
    y_true: np.typing.ArrayLike, y_pred: np.typing.ArrayLike
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title("Confusion Matrix", pad=16)

    plt.tight_layout()

    return fig


def get_roc_curve_fig(
    y_true: np.typing.NDArray[np.float64],
    y_proba: np.typing.NDArray[np.float64],
    encoder: LabelEncoder,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))

    n_classes = encoder.classes_.shape[0]
    y_test_bin = np.asarray(
        label_binarize(y_true, classes=range(n_classes)), dtype=np.float64
    )

    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_proba[:, i],
            name=f"{encoder.classes_[i]}",
            ax=ax,
        )

    ax.set_title("ROC Curve", pad=16)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    plt.tight_layout()

    return fig


def get_precision_recall_curve_fig(
    y_true: np.typing.NDArray[np.float64],
    y_proba: np.typing.NDArray[np.float64],
    encoder: LabelEncoder,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))

    n_classes = encoder.classes_.shape[0]
    y_test_bin = np.asarray(
        label_binarize(y_true, classes=range(n_classes)), dtype=np.float64
    )

    for i in range(n_classes):
        PrecisionRecallDisplay.from_predictions(
            y_test_bin[:, i],
            y_proba[:, i],
            name=f"{encoder.classes_[i]}",
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
