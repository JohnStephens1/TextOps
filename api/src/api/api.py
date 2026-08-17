import datetime

import pandas as pd
from fastapi import FastAPI

from text_classifier.data.data_pipe import raw_to_model_input_pipe
from text_classifier.mlflow_loader import get_champ_model_encoder_emb_model

from .schema import PredictionRequest, PredictionResponse

api = FastAPI()

model, label_encoder, embedding_model = get_champ_model_encoder_emb_model()


def get_date_time() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def get_pred_response(model_input: pd.DataFrame) -> PredictionResponse:
    preds = model.predict(model_input)
    preds_proba = model.predict_proba(model_input)

    certainties = preds_proba.max(axis=1)

    labels = label_encoder.inverse_transform(preds)
    all_labels = label_encoder.classes_

    return PredictionResponse(
        pred=preds.tolist()[0],
        pred_proba=preds_proba.tolist()[0],
        certainty=certainties.tolist()[0],
        label=labels.tolist()[0],
        all_labels=all_labels.tolist()[0],
    )


@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model_input = raw_to_model_input_pipe(
        embedding_model, request.title, request.description, get_date_time()
    )

    return get_pred_response(model_input)
