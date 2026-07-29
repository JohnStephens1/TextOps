from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainingData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.typing.NDArray[np.int64]
    y_test: np.typing.NDArray[np.int64]


@dataclass
class Artifacts:
    pipe: Pipeline
    label_encoder: LabelEncoder


@dataclass
class TrainingContext:
    dataset: TrainingData
    artifacts: Artifacts


@dataclass
class Predictions:
    y_true: np.typing.NDArray[np.int64]
    y_pred: np.typing.NDArray[np.int64]
    y_proba: np.typing.NDArray[np.float64]


@dataclass
class PredictionsEncoder:
    predictions: Predictions
    encoder: LabelEncoder
