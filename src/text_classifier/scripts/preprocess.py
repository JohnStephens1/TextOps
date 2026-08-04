import logging

from text_classifier.config.config import PREPROCESSED_DATASET_PATH
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.dataset import load_raw_dataset, save_dataset
from text_classifier.data.preprocessing import preprocess_df

setup_logging()


logger = logging.getLogger(__name__)


def main() -> None:
    """loads the raw_dataset, preprocesses it, then saves it to parquet"""

    logger.info("Preprocessing...")

    df = load_raw_dataset()
    df = preprocess_df(df)
    save_dataset(df, PREPROCESSED_DATASET_PATH)

    logger.info("Saved preprocessed dataset")


if __name__ == "__main__":
    main()
