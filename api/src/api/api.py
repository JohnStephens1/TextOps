from contextlib import asynccontextmanager

from fastapi import FastAPI

from text_classifier.mlflow_loader import get_champ_model_encoder_emb_model

from .dependencies import PredictionResourcesDeps
from .model_io import get_model_input, get_pred_response
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


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    resources: PredictionResourcesDeps,
) -> PredictionResponse:
    model_input = get_model_input(request, resources)
    pred_response = get_pred_response(resources, model_input)

    return pred_response
