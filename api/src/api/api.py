from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from text_classifier.mlflow_loader import get_champ_model_prediction_resources

from .dependencies import PredictionResourcesDeps
from .model_io import get_model_input, get_pred_response
from .schema import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.prediction_resources = get_champ_model_prediction_resources()

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    resources: PredictionResourcesDeps,
) -> PredictionResponse:
    model_input = get_model_input(request, resources)
    pred_response = get_pred_response(resources, model_input)

    return pred_response
