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
from text_classifier.schema import TrainTestSplits


def save_data_splits(splits: TrainTestSplits) -> None:
    save(splits.X_train, X_TRAIN_PATH)
    save(splits.X_test, X_TEST_PATH)
    save(splits.y_train, Y_TRAIN_PATH)
    save(splits.y_test, Y_TEST_PATH)


def save_label_encoder(label_encoder: LabelEncoder) -> None:
    save(label_encoder, ENCODER_ARTIFACT_PATH)


def get_X_y(
    df: pd.DataFrame, target_col: str = "tag"
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y


def encode_y(
    y: pd.Series, target_col: str = "tag"
) -> tuple[LabelEncoder, pd.DataFrame]:
    """returns the used label encoder and the encoded y"""

    label_encoder = LabelEncoder()

    y_encoded = np.asarray(label_encoder.fit_transform(y), dtype=np.int64)
    y_encoded = pd.DataFrame(
        y_encoded, index=y.index, columns=[target_col], dtype=np.int64
    )

    return label_encoder, y_encoded


def get_train_test_splits(
    X: pd.DataFrame,
    y_encoded: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 1234,
) -> TrainTestSplits:
    # TODO could extract split config | save split metadata

    return TrainTestSplits(
        *train_test_split(
            X, y_encoded, stratify=y_encoded, test_size=test_size, random_state=seed
        )
    )


def get_train_test_splits_encoder(
    model_df: pd.DataFrame,
) -> tuple[TrainTestSplits, LabelEncoder]:
    X, y = get_X_y(model_df)

    label_encoder, y_encoded = encode_y(y)
    splits = get_train_test_splits(X, y_encoded)

    return splits, label_encoder
