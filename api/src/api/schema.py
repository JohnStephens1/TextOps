from pydantic import BaseModel


class PredictionRequest(BaseModel):
    title: str
    description: str


class PredictionResponse(BaseModel):
    preds: list[float]
    preds_proba: list[list[float]]
    certainties: list[float]
    labels: list[str]
    all_labels: list[str]
