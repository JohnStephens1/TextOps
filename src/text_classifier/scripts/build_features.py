from text_classifier.config.config import (
    FEATURE_DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
)
from text_classifier.data.dataset import load_dataset, save_dataset
from text_classifier.data.features import add_features


def main() -> None:
    """loads the preprocessed dataset, adds features, then saves it to parquet"""

    df = load_dataset(PREPROCESSED_DATASET_PATH)
    df = add_features(df)
    save_dataset(df, FEATURE_DATASET_PATH)


if __name__ == "__main__":
    main()
