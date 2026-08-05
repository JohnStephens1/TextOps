import logging

from text_classifier.config.config import ENCODER_ARTIFACT_PATH, MODEL_DATASET_PATH
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.split import (
    get_train_test_splits_encoder,
    save_data_splits,
)
from text_classifier.save_load import load_parquet, save

setup_logging()


logger = logging.getLogger("text_classifier.scripts.build_train_test_datasets")


def main() -> None:
    """loads the model dataset, prepares and splits it into train and test sets, then saves the results to parquet.

    In detail:
    - loads model dataset
    - splits into X, y
    - encodes y
    - splits into train and test sets
    - saves train and test datasets
    - saves label encoder
    """

    logger.info("Building train and test datasets...")

    df = load_parquet(MODEL_DATASET_PATH)
    train_test_splits, label_encoder = get_train_test_splits_encoder(df)

    save_data_splits(train_test_splits)
    logger.info("Saved train and test splits")

    save(label_encoder, ENCODER_ARTIFACT_PATH)
    logger.info("Saved label encoder")


if __name__ == "__main__":
    main()
