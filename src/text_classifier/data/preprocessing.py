import pandas as pd


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def set_date_time(df: pd.DataFrame) -> pd.DataFrame:
    df["created_on"] = pd.to_datetime(df["created_on"])

    return df


def preprocess_string(string: str) -> str:
    string = string.lower().strip()

    return string


def preprocess_text(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    df[target_cols] = df[target_cols].apply(lambda col: col.apply(preprocess_string))

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = set_date_time(df)
    df = preprocess_text(df, target_cols=["title", "description"])

    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_text(df, target_cols=["tag"])

    return df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_df(df)
    df = preprocess_features(df)
    df = preprocess_target(df)

    return df
