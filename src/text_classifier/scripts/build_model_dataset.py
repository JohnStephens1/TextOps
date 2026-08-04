import logging

from text_classifier.config.config import (
    FEATURE_DATASET_PATH,
    MODEL_DATASET_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.dataset import load_dataset, save_dataset
from text_classifier.data.model_data import prepare_model_data

setup_logging()


logger = logging.getLogger("text_classifier.scripts.build_model_dataset")


def main() -> None:
    """loads the feature dataset, prepares it for model input, then saves it to parquet"""

    logger.info("Building model dataset...")

    df = load_dataset(FEATURE_DATASET_PATH)
    df = prepare_model_data(df)
    save_dataset(df, MODEL_DATASET_PATH)

    logger.info("Saved model dataset")


if __name__ == "__main__":
    main()
