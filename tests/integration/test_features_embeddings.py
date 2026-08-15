from text_classifier.config.config import PREPROCESSED_DATASET_PATH
from text_classifier.data.features import add_features
from text_classifier.save_load import load_parquet


def test_identical_output_for_cached_and_new_embeddings() -> None:
    df = load_parquet(PREPROCESSED_DATASET_PATH)

    df_cached_embs = add_features(df, regenerate_embs=False)
    df_new_embs = add_features(df, regenerate_embs=True)

    assert df_cached_embs.equals(df_new_embs)
