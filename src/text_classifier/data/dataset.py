import pandas as pd

from text_classifier.config.config import DATASET_PATH


def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)
