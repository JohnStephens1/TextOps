import logging

from text_classifier.config.config import (
    FEATURE_DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
)
from text_classifier.config.logger import setup_logger
from text_classifier.data.dataset import load_dataset, save_dataset
from text_classifier.data.features import add_features

setup_logger()


logger = logging.getLogger(__name__)


def main() -> None:
    """loads the preprocessed dataset, adds features, then saves it to parquet"""

    logger.info("Building features...")

    df = load_dataset(PREPROCESSED_DATASET_PATH)
    df = add_features(df)
    save_dataset(df, FEATURE_DATASET_PATH)

    logger.info("Saved feature dataset")


if __name__ == "__main__":
    main()
