from pathlib import Path
from typing import Any

import joblib  # type: ignore
import pandas as pd


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


def _save_joblib(obj: Any, path: Path) -> None:
    joblib.dump(obj, path)


def _save_text(string: str, path: Path) -> None:
    path.write_text(string)


_SAVERS = {
    ".parquet": _save_parquet,
    ".joblib": _save_joblib,
    ".txt": _save_text,
}


def save(obj: Any, path: Path) -> None:
    try:
        saver = _SAVERS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    path.parent.mkdir(exist_ok=True, parents=True)

    saver(obj, path)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def load_text(path: Path) -> str:
    return path.read_text()
