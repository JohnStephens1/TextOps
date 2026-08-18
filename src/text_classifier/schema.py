from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from text_classifier.protocols import Predictor


@dataclass
class XYData:
    X: pd.DataFrame
    y: np.typing.NDArray[np.int64]


@dataclass
class TrainTestSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


@dataclass
class Artifacts:
    pipe: Pipeline
    label_encoder: LabelEncoder


@dataclass
class Predictions:
    y_true: np.typing.NDArray[np.int64]
    y_pred: np.typing.NDArray[np.int64]
    y_proba: np.typing.NDArray[np.float64]


@dataclass
class PredictionsEncoder:
    predictions: Predictions
    encoder: LabelEncoder


@dataclass(frozen=True)
class RawModelInput:
    title: str
    description: str
    date_time: datetime


@dataclass
class PredictionResources:
    model: Predictor
    label_encoder: LabelEncoder
    embedding_model: SentenceTransformer
