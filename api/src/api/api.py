import datetime
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Request

from text_classifier.data.data_pipe import raw_to_model_input_pipe
from text_classifier.mlflow_loader import get_champ_model_encoder_emb_model
from text_classifier.schema import RawModelInput

from .schema import PredictionRequest, PredictionResources, PredictionResponse

# TODO clean, delegate, extract


def load_prediction_resources() -> PredictionResources:
    model, label_encoder, embedding_model = get_champ_model_encoder_emb_model()

    return PredictionResources(
        model=model,
        label_encoder=label_encoder,
        embedding_model=embedding_model,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_resources = load_prediction_resources()

    yield


app = FastAPI(lifespan=lifespan)


def get_current_date_time() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


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


def get_prediction_resources(request: Request) -> PredictionResources:
    return request.app.state.prediction_resources


PredictionResourcesDeps = Annotated[
    PredictionResources, Depends(get_prediction_resources)
]


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


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    resources: PredictionResourcesDeps,
) -> PredictionResponse:
    model_input = get_model_input(request, resources)
    pred_response = get_pred_response(resources, model_input)

    return pred_response
