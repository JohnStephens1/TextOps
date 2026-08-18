import pandas as pd
from sentence_transformers import SentenceTransformer

from text_classifier.data.embeddings import add_text_embeddings
from text_classifier.data.features import add_features
from text_classifier.data.model_data import drop_non_feature_cols
from text_classifier.data.preprocessing import preprocess_features
from text_classifier.schema import RawModelInput


def get_df_from_input(raw_model_input: RawModelInput) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "created_on": [raw_model_input.date_time],
            "title": [raw_model_input.title],
            "description": [raw_model_input.description],
        }
    )


def raw_to_model_input_pipe(
    embedding_model: SentenceTransformer,
    raw_model_input: RawModelInput,
) -> pd.DataFrame:
    df = get_df_from_input(raw_model_input)
    df = preprocess_features(df)
    df = add_features(df)
    df = add_text_embeddings(df, embedding_model, regenerate=True)
    df = drop_non_feature_cols(df)

    return df
