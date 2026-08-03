from text_classifier.config.config import PREPROCESSED_DATASET_PATH
from text_classifier.data.dataset import load_raw_dataset, save_dataset
from text_classifier.data.preprocessing import preprocess_df


def main() -> None:
    """loads the raw_dataset, preprocesses it, then saves it to parquet"""

    df = load_raw_dataset()
    df = preprocess_df(df)
    save_dataset(df, PREPROCESSED_DATASET_PATH)


if __name__ == "__main__":
    main()
