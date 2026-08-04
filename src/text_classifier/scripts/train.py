from text_classifier.config.config import MODEL_DATASET_PATH
from text_classifier.data.dataset import load_dataset
from text_classifier.model.training import train_from_config


def main() -> None:
    df = load_dataset(MODEL_DATASET_PATH)

    # encoder, train_data, my_model, metrics, figs
    _, _, _, _, _ = train_from_config(df)


if __name__ == "__main__":
    main()
