import logging

from text_classifier.config.config import MODEL_DATASET_PATH
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.dataset import load_dataset
from text_classifier.model.training import train_from_config

setup_logging()


logger = logging.getLogger("text_classifier.scripts.train")


def main() -> None:
    """trains, evaluates and tracks model experiment end-to-end.

    In detail:
    - loads the model dataset
    - prepares it for training (label_encoding, X, y, train, test)
    - trains model from params.yaml config
    - creates metrics, plots for trained model
    - tracks model, metrics, plots using MLFlow
    - generates train_metadata.json with experiment info
    """

    logger.info("Setting up training...")

    df = load_dataset(MODEL_DATASET_PATH)

    _, _, _, _, _ = train_from_config(df)

    logger.info("Training completed")


if __name__ == "__main__":
    main()
