import logging

from sentence_transformers import SentenceTransformer

from text_classifier.config.config import (
    EMBEDDING_DATASET_PATH,
    EMBEDDING_MODEL_STR,
    EMBEDDING_MODEL_STR_ARTIFACT_PATH,
    FEATURE_DATASET_PATH,
)
from text_classifier.config.logging_config import setup_logging
from text_classifier.data.embeddings import add_text_embeddings
from text_classifier.save_load import load_parquet, save

setup_logging()


logger = logging.getLogger("text_classifier.scripts.build_embeddings")


def main() -> None:
    """loads the preprocessed dataset and embedding model, adds embeddings, then saves dataset to parquet, embedding model string to text"""

    logger.info("Building embeddings...")

    df = load_parquet(FEATURE_DATASET_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_STR)

    df = add_text_embeddings(df, embedding_model)

    save(EMBEDDING_MODEL_STR, EMBEDDING_MODEL_STR_ARTIFACT_PATH)
    logger.info("Saved embedding model string")

    save(df, EMBEDDING_DATASET_PATH)
    logger.info("Saved embedding dataset")


if __name__ == "__main__":
    main()
