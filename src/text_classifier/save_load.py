from pathlib import Path
from typing import Any

import joblib  # type: ignore
from pandas import DataFrame


def _save_parquet(df: DataFrame, path: Path):
    df.to_parquet(path)


def _save_joblib(obj: Any, path: Path):
    joblib.dump(obj, path)


def _save_text(string: str, path: Path):
    path.write_text(string)


_SAVERS = {
    ".parquet": _save_parquet,
    ".joblib": _save_joblib,
    ".txt": _save_text,
}


def save(obj: Any, path: Path):
    try:
        saver = _SAVERS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    path.parent.mkdir(exist_ok=True, parents=True)

    saver(obj, path)
