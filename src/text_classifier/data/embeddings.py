import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from text_classifier.config.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_STR,
    EMBEDDINGS_PATH,
)


def save_embeddings(
    ids: np.typing.ArrayLike,
    embeddings: np.typing.ArrayLike,
    file_path: Path = EMBEDDINGS_PATH,
    model: str = EMBEDDING_MODEL_STR,
) -> None:
    """saves embeddings to file_path, including ids and model name

    Args:
        ids (np.typing.ArrayLike): ids array
        embeddings (np.typing.ArrayLike): embeddings array
        file_path (Path): file path to save location
        model (str, optional): used embedding model name. Defaults to EMBEDDING_MODEL_STR.
    """
    np.savez(file_path, ids=ids, embeddings=embeddings, model=model)


def load_ids_embeddings(
    file_path: Path = EMBEDDINGS_PATH,
    embedding_dim: int = EMBEDDING_DIM,
) -> tuple[np.ndarray, np.typing.NDArray[np.float32]]:
    """loads the ids and embeddings stored in file_path

    Args:
        file_path (Path): file path to stored data

    Returns:
        tuple[np.ndarray, np.ndarray]: ids, embeddings
    """
    if not os.path.exists(file_path):
        return np.empty(0), np.empty((0, embedding_dim), dtype=np.float32)

    loaded_file = np.load(file_path)

    return loaded_file["ids"], loaded_file["embeddings"]


def get_intersecting_complementing_ids(
    df: pd.DataFrame, ids_loaded: np.ndarray
) -> tuple[pd.Index, pd.Index]:
    """gets the intersection between df and saved ids, as well as the remaining ids that need to be generated

    Args:
        df (pd.DataFrame): input df
        ids_loaded (np.ndarray): the loaded ids from saved embeddings

    Returns:
        tuple[pd.Index, pd.Index]: intersecting_ids, ids_to_generate.
        intersection between df and saved ids, as well as the remaining ids that need to be generated
    """
    # all ids in input df and embeddings -> to load
    intersecting_ids = df.index.intersection(ids_loaded.tolist())

    # all ids - intersecting ids -> to generate
    ids_to_generate = df.index.difference(intersecting_ids)

    return intersecting_ids, ids_to_generate


def get_new_embeddings(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    ids_to_generate: pd.Index,
    text_col: str = "text",
) -> np.typing.NDArray[np.float32]:
    """generates new embeddings for df based on ids_to_generate using the model defined in model_str

    Args:
        df (pd.DataFrame): input df
        embedding_model (SentenceTransformer): embedding model
        ids_to_generate (pd.Index): corresponding ids to generate
        text_col (str): name of the text column
        model_str (str, optional): name of the embedding model. Defaults to EMBEDDING_MODEL_STR.

    Returns:
        np.ndarray: generated embeddings
    """
    text_to_generate = df.loc[ids_to_generate][text_col].to_list()

    new_embeddings = embedding_model.encode(text_to_generate, convert_to_numpy=True)

    return new_embeddings


def get_and_save_generated_embeddings(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    ids_loaded: np.ndarray,
    ids_to_generate: pd.Index,
    embeddings_loaded: np.ndarray,
    file_path: Path = EMBEDDINGS_PATH,
    embedding_dim: int = EMBEDDING_DIM,
) -> np.typing.NDArray[np.float32]:
    """gets generated embeddings, loading and saving as needed, preventing unnecessary generation while automatically saving newly generated embeddings

    Args:
        df (pd.DataFrame): input df
        embedding_model (SentenceTransformer): embedding model
        ids_loaded (np.ndarray): loaded ids
        ids_to_generate (pd.Index): ids of rows for embedding generation
        embeddings_loaded (np.ndarray): loaded embeddings
        file_path (Path, optional): path to the stored embeddings. Defaults to EMBEDDINGS_PATH.
        embedding_dim (int, optional): output dimension of the used embedding. Defaults to EMBEDDING_DIM.

    Returns:
        np.ndarray: the generated embeddings
    """
    if ids_to_generate.empty:
        return np.empty((0, embedding_dim), dtype=np.float32)

    # generate embeddings for ids_to_generate
    embeddings_generated = get_new_embeddings(df, embedding_model, ids_to_generate)

    # to_save = loaded + generated
    ids_to_save = np.concat([ids_loaded, ids_to_generate])
    embeddings_to_save = np.concat([embeddings_loaded, embeddings_generated])

    save_embeddings(ids_to_save, embeddings_to_save, file_path)

    return embeddings_generated


def get_intersecting_embeddings(
    ids: np.ndarray,
    embeddings: np.ndarray,
    ids_intersecting: pd.Index,
    embedding_dim: int = EMBEDDING_DIM,
) -> np.typing.NDArray[np.float32]:
    """gets all embeddings with ids in ids_intersecting

    Args:
        ids (np.ndarray): ids corresponding to embeddings
        embeddings (np.ndarray): embeddings corresponding to ids
        ids_intersecting (pd.Index): ids that must be present in both sets to include corresponding embeddings
        embedding_dim (int, optional): output dimension of the used embedding. Defaults to EMBEDDING_DIM.

    Returns:
        np.ndarray: intersecting embeddings
    """
    return np.array(
        [emb for id, emb in zip(ids, embeddings) if id in ids_intersecting],
        dtype=np.float32,
    ).reshape(-1, embedding_dim)


def get_embeddings_df(
    ids: pd.Index,
    embeddings: np.ndarray,
    text_col: str,
    embedding_dim: int = EMBEDDING_DIM,
) -> pd.DataFrame:
    """gets a df containing input embeddings with ids as index

    Args:
        ids (pd.Index): index ids
        embeddings (np.ndarray): embeddings
        text_col (str): name of the output column name
        embedding_dim (int, optional): output dimension of the used embedding. Defaults to EMBEDDING_DIM.

    Returns:
        pd.DataFrame: embeddings_df
    """
    return pd.DataFrame(
        embeddings,
        index=ids,
        columns=[f"{text_col}_{i}" for i in range(embedding_dim)],
    )


def add_text_embeddings_w_cache(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    text_col: str = "text",
    file_path: Path = EMBEDDINGS_PATH,
    embedding_dim: int = EMBEDDING_DIM,
) -> pd.DataFrame:
    """adds text embeddings to the input_df, saving and loading embeddings as needed

    Args:
        df (pd.DataFrame): input df
        embedding_model (SentenceTransformer): embedding model
        text_col (str, optional): name of the column containing the text to be transformed. Defaults to "text".
        file_path (Path, optional): path to the stored embeddings. Defaults to EMBEDDINGS_PATH.
        embedding_dim (int, optional): output dimension of the used embedding. Defaults to EMBEDDING_DIM.

    Returns:
        pd.DataFrame: modified df
    """
    ids_loaded, embeddings_loaded = load_ids_embeddings(file_path)

    ids_intersecting, ids_to_generate = get_intersecting_complementing_ids(
        df, ids_loaded
    )

    embeddings_generated = get_and_save_generated_embeddings(
        df, embedding_model, ids_loaded, ids_to_generate, embeddings_loaded
    )

    embeddings_intersecting = get_intersecting_embeddings(
        ids_loaded, embeddings_loaded, ids_intersecting
    )

    # result = intersecting + generated
    ids_result = np.concat([ids_intersecting, ids_to_generate])
    embeddings_result = np.concat([embeddings_intersecting, embeddings_generated])

    # create embeddings df for result
    df_embeddings = pd.DataFrame(
        embeddings_result,
        index=ids_result,
        columns=[f"{text_col}_{i}" for i in range(embedding_dim)],
        dtype=np.float32,
    )

    # merge into input df
    df_result = pd.concat([df, df_embeddings], axis=1)

    return df_result


def add_new_text_embeddings(
    df: pd.DataFrame, embedding_model: SentenceTransformer, text_col: str = "text"
) -> pd.DataFrame:
    embeddings = embedding_model.encode(df[text_col].to_list(), convert_to_numpy=True)

    df_embeddings = pd.DataFrame(
        embeddings,
        index=df.index,
        columns=[f"{text_col}_{i}" for i in range(embeddings.shape[1])],
        dtype=np.float32,
    )

    df = pd.concat([df, df_embeddings], axis=1)

    return df


def add_text_embeddings(
    df: pd.DataFrame, embedding_model: SentenceTransformer, regenerate: bool = False
) -> pd.DataFrame:
    df = (
        add_new_text_embeddings(df, embedding_model)
        if regenerate
        else add_text_embeddings_w_cache(df, embedding_model)
    )

    return df
