import pandas as pd


def drop_non_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["created_on", "title", "description", "text"]

    df = df.drop(drop_cols, axis=1)

    return df


def prepare_model_data(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = drop_non_feature_cols(feature_df)

    return df
