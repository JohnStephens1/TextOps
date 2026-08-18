import datetime
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

from text_classifier.protocols import Predictor


class PredictionRequest(BaseModel):
    title: str
    description: str


class PredictionResponse(BaseModel):
    pred: float
    pred_proba: list[float]
    certainty: float
    label: str
    all_labels: list[str]


@dataclass
class PredictionResources:
    model: Predictor
    label_encoder: LabelEncoder
    embedding_model: SentenceTransformer


class RawModelInput(BaseModel):
    title: str
    description: str
    date_time: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )

    @classmethod
    def from_request(cls, request: PredictionRequest) -> Self:
        return cls(
            title=request.title,
            description=request.description,
        )
