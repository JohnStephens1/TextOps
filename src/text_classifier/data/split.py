from dataclasses import dataclass

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainTestSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


def split_save_model_df_encoder(model_df: pd.DataFrame):
    target_col = "tag"
    test_size = 0.2
    seed = 1234

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

    # TODO UPDATE SAVER, PATHS
    splits.X_train.to_parquet("splits/X_train.parquet")
    splits.X_test.to_parquet("splits/X_test.parquet")
    splits.y_train.to_parquet("splits/y_train.parquet")
    splits.y_test.to_parquet("splits/y_test.parquet")

    # for field in fields(splits):
    #     data = getattr(splits, field.name)
    #     data.to_parquet(f"{field.name}.parquet")

    joblib.dump(label_encoder, "artifacts/label_encoder.joblib")
