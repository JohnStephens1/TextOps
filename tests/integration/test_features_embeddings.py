from sentence_transformers import SentenceTransformer

from text_classifier.config.config import (
    EMBEDDING_MODEL_STR,
    FEATURE_DATASET_PATH,
)
from text_classifier.data.embeddings import add_text_embeddings
from text_classifier.save_load import load_parquet


def test_identical_output_for_cached_and_new_embeddings() -> None:
    df = load_parquet(FEATURE_DATASET_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_STR)

    df_cached_embs = add_text_embeddings(df, embedding_model, regenerate=False)
    df_new_embs = add_text_embeddings(df, embedding_model, regenerate=True)

    assert df_cached_embs.equals(df_new_embs)
