import logging

from text_classifier.config.config import (
    FEATURE_DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.dataset import load_dataset, save_dataset
from text_classifier.data.features import add_features

setup_logging()


logger = logging.getLogger("text_classifier.scripts.build_features")


def main() -> None:
    """loads the preprocessed dataset, adds features, then saves it to parquet"""

    logger.info("Building features...")

    df = load_dataset(PREPROCESSED_DATASET_PATH)
    df = add_features(df)
    save_dataset(df, FEATURE_DATASET_PATH)

    logger.info("Saved feature dataset")


if __name__ == "__main__":
    main()
