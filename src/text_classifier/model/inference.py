import logging
from datetime import datetime

import numpy as np
import pandas as pd

from text_classifier.config.logging_config import setup_logging
from text_classifier.data.data_pipe import raw_to_model_input_pipe
from text_classifier.mlflow_loader import get_champ_model_encoder_emb_model

setup_logging()


logger = logging.getLogger("text_classifier.model.inference")


# def fancy_schmancy_sort_key(col: str) -> tuple[int, int | str]:
#     """sorts columns alphabetically, aside from text_<int> columns, which come last. text_<int> columns are sorted numerically, based on their <int>."""
#     splits = col.split("_")

#     if splits[0] == "text" and splits[-1].isdigit():
#         return (1, int(splits[-1]))
#     else:
#         return (0, col)


def get_df_from_input(title: str, description: str, date_time: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "created_on": [date_time],
            "title": [title],
            "description": [description],
        }
    )


def get_current_date_time() -> datetime:
    return datetime.now().astimezone()


# date_time: str | datetime ?
def inference(
    title: str, description: str, date_time: str
) -> tuple[np.ndarray, np.ndarray]:
    # TODO add pydantic check

    model, label_encoder, embedding_model = get_champ_model_encoder_emb_model()
    df = raw_to_model_input_pipe(embedding_model, title, description, date_time)

    preds_proba = model.predict_proba(df)
    preds = model.predict(df)
    labels = label_encoder.inverse_transform(preds)

    print(f"""
        Input:
        - title: {title}
        - description: {description}
        Output:
        - possible labels: {label_encoder.classes_}
        - certainty: {[f"{x:.4f}" for x in preds_proba[0]]}
        - prediction: {labels}
    """)

    return preds_proba, labels


# print(inference("langue", "fabricating and refining", str(get_current_date_time())))
