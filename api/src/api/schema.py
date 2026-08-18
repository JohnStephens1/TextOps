from dataclasses import dataclass

from pydantic import BaseModel
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
