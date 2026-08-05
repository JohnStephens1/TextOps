import logging

from text_classifier.config.config import PREPROCESSED_DATASET_PATH, RAW_DATASET_PATH
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.preprocessing import preprocess_df
from text_classifier.save_load import load_csv, save

setup_logging()


logger = logging.getLogger("text_classifier.scripts.preprocess")


def main() -> None:
    """loads the raw_dataset, preprocesses it, then saves it to parquet"""

    logger.info("Preprocessing...")

    df = load_csv(RAW_DATASET_PATH)
    df = df.set_index("id")
    df = preprocess_df(df)

    save(df, PREPROCESSED_DATASET_PATH)
    logger.info("Saved preprocessed dataset")


if __name__ == "__main__":
    main()
