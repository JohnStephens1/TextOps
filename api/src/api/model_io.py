import pandas as pd

from text_classifier.data.data_pipe import raw_to_model_input_pipe
from text_classifier.schema import RawModelInput

from .datetime_utils import get_current_date_time
from .schema import PredictionRequest, PredictionResources, PredictionResponse


def get_raw_model_input_from_request(request: PredictionRequest) -> RawModelInput:
    return RawModelInput(
        title=request.title,
        description=request.description,
        date_time=get_current_date_time(),
    )


def get_model_input(
    request: PredictionRequest,
    resources: PredictionResources,
) -> pd.DataFrame:
    raw_model_input = get_raw_model_input_from_request(request)
    model_input = raw_to_model_input_pipe(resources.embedding_model, raw_model_input)

    return model_input


def get_pred_response(
    resources: PredictionResources,
    model_input: pd.DataFrame,
) -> PredictionResponse:
    preds = resources.model.predict(model_input)
    preds_proba = resources.model.predict_proba(model_input)

    certainties = preds_proba.max(axis=1)

    labels = resources.label_encoder.inverse_transform(preds)
    all_labels = resources.label_encoder.classes_

    return PredictionResponse(
        pred=preds.tolist()[0],
        pred_proba=preds_proba.tolist()[0],
        certainty=certainties.tolist()[0],
        label=labels.tolist()[0],
        all_labels=all_labels.tolist(),
    )
