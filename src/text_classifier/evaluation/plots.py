import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import ConfusionMatrixDisplay


def get_confusion_matrix_fig(y_true: ArrayLike, y_pred: ArrayLike):
    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(
        "Confusion Matrix",
        pad=16,
    )

    plt.tight_layout()

    return fig
