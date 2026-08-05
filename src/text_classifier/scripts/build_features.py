import logging

from text_classifier.config.config import (
    FEATURE_DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.features import add_features
from text_classifier.save_load import load_parquet, save

setup_logging()


logger = logging.getLogger("text_classifier.scripts.build_features")


def main() -> None:
    """loads the preprocessed dataset, adds features (text, time series, embeddings), then saves it to parquet"""

    logger.info("Building features...")

    df = load_parquet(PREPROCESSED_DATASET_PATH)
    df = add_features(df)
    save(df, FEATURE_DATASET_PATH)

    logger.info("Saved feature dataset")


if __name__ == "__main__":
    main()
