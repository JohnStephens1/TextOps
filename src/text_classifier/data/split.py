from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import (
    ENCODER_ARTIFACT_PATH,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)
from text_classifier.save_load import save


@dataclass
class TrainTestSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


def save_data_splits(splits: TrainTestSplits) -> None:
    save(splits.X_train, X_TRAIN_PATH)
    save(splits.X_test, X_TEST_PATH)
    save(splits.y_train, Y_TRAIN_PATH)
    save(splits.y_test, Y_TEST_PATH)


def save_label_encoder(label_encoder: LabelEncoder) -> None:
    save(label_encoder, ENCODER_ARTIFACT_PATH)


def get_train_test_splits_encoder(
    model_df: pd.DataFrame,
) -> tuple[TrainTestSplits, LabelEncoder]:
    target_col = "tag"
    test_size = 0.2
    seed = 1234
    # TODO could extract split config | save split metadata

    X = model_df.drop(target_col, axis=1)
    y = model_df[target_col]

    label_encoder = LabelEncoder()

    y_encoded = np.asarray(label_encoder.fit_transform(y), dtype=np.int64)
    y_encoded = pd.DataFrame(
        y_encoded, index=y.index, columns=[target_col], dtype=np.int64
    )

    splits = TrainTestSplits(
        *train_test_split(
            X, y_encoded, stratify=y, test_size=test_size, random_state=seed
        )
    )

    return splits, label_encoder
