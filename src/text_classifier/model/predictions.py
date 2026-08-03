import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

from text_classifier.protocols import Predictor
from text_classifier.schema import Predictions, PredictionsEncoder


def get_predictions(
    model: Predictor, X: pd.DataFrame, y: NDArray[np.int64]
) -> Predictions:
    return Predictions(
        y_true=y,
        y_pred=model.predict(X),
        y_proba=model.predict_proba(X),
    )


def get_predictions_w_encoder_from_predictions(
    predictions: Predictions, label_encoder: LabelEncoder
) -> PredictionsEncoder:
    return PredictionsEncoder(
        predictions=predictions,
        encoder=label_encoder,
    )


def get_predictions_w_encoder(
    model: Predictor,
    X: pd.DataFrame,
    y: NDArray[np.int64],
    label_encoder: LabelEncoder,
) -> PredictionsEncoder:
    predictions = get_predictions(model, X, y)

    return PredictionsEncoder(
        predictions=predictions,
        encoder=label_encoder,
    )
