from pydantic import BaseModel


class PredictionRequest(BaseModel):
    title: str
    description: str


class PredictionResponse(BaseModel):
    pred: float
    pred_proba: list[float]
    certainty: float
    label: str
    all_labels: list[str]
