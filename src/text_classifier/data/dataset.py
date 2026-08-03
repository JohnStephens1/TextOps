import pandas as pd

from text_classifier.config.config import RAW_DATASET_PATH


def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET_PATH)
