from pathlib import Path

from text_classifier.save_load import load_parquet
from text_classifier.schema import XYData


def get_x_y_data(X_path: Path, y_path: Path) -> XYData:
    return XYData(
        load_parquet(X_path),
        load_parquet(y_path).iloc[:, 0].to_numpy(),
    )
