# from pathlib import Path

# import pandas as pd

# from text_classifier.config.config import RAW_DATASET_PATH

# # TODO implement save_load.py over this


# def load_raw_dataset() -> pd.DataFrame:
#     df = pd.read_csv(RAW_DATASET_PATH)
#     df = df.set_index("id")

#     return df


# def load_dataset(path: Path) -> pd.DataFrame:
#     return pd.read_parquet(path)


# def save_dataset(df: pd.DataFrame, path: Path) -> None:
#     df.to_parquet(path)
