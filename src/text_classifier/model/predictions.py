from sklearn.preprocessing import LabelEncoder

from text_classifier.protocols import Predictor
from text_classifier.schema import Predictions, PredictionsEncoder, XYData


def get_predictions(model: Predictor, ds: XYData) -> Predictions:
    return Predictions(
        y_true=ds.y,
        y_pred=model.predict(ds.X),
        y_proba=model.predict_proba(ds.X),
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
    ds: XYData,
    label_encoder: LabelEncoder,
) -> PredictionsEncoder:
    predictions = get_predictions(model, ds)

    return PredictionsEncoder(
        predictions=predictions,
        encoder=label_encoder,
    )
