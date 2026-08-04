from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from text_classifier.config.config import RAW_DATASET_PATH
from text_classifier.data.features import add_features
from text_classifier.data.model_data import prepare_model_data
from text_classifier.data.preprocessing import preprocess_df
from text_classifier.data.train_data import get_encoder_train_data
from text_classifier.schema import TrainingData


def get_raw_dataset(ds_path: Path | None = None) -> pd.DataFrame:
    """gets the raw dataset

    Args:
        ds_path (Path, optional): path to the dataset. Defaults to DATASET_PATH.

    Returns:
        pd.DataFrame: the loaded df
    """
    if ds_path is None:
        ds_path = RAW_DATASET_PATH

    return pd.read_csv(ds_path)


def data_pipeline() -> pd.DataFrame:
    """preprocesses the data and adds features to the dataset

    Returns:
        pd.DataFrame: the prepared df
    """
    df = get_raw_dataset()
    df = df.set_index("id")

    df = preprocess_df(df)
    df = add_features(df)

    return df


def model_data_pipeline() -> tuple[LabelEncoder, TrainingData]:
    df = data_pipeline()
    df = prepare_model_data(df)

    encoder, train_data = get_encoder_train_data(df)

    return encoder, train_data
