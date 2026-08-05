import logging

from text_classifier.config.config import (
    FEATURE_DATASET_PATH,
    MODEL_DATASET_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.model_data import prepare_model_data
from text_classifier.save_load import load_parquet, save

setup_logging()


logger = logging.getLogger("text_classifier.scripts.build_model_dataset")


def main() -> None:
    """loads the feature dataset, prepares it for model input (dropping non-feature columns), then saves it to parquet"""

    logger.info("Building model dataset...")

    df = load_parquet(FEATURE_DATASET_PATH)
    df = prepare_model_data(df)
    save(df, MODEL_DATASET_PATH)

    logger.info("Saved model dataset")


if __name__ == "__main__":
    main()
