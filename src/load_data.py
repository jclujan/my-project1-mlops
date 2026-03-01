"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoadError(RuntimeError):
    """Raised when data cannot be loaded properly."""


@dataclass(frozen=True)
class LoadResult:
    """Container for loaded datasets."""
    df: pd.DataFrame
    source: str


def load_csv(path: str | Path, *, encoding: str = "utf-8") -> LoadResult:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file.
        encoding: File encoding.

    Returns:
        LoadResult containing the dataframe and a source string.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataLoadError: If the CSV is empty or unreadable.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise DataLoadError(f"Expected a .csv file, got: {file_path.suffix}")

    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as exc:  # keep broad here; raise a clearer message
        raise DataLoadError(f"Failed to read CSV at {file_path}: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"CSV loaded but contains no rows: {file_path}")

    return LoadResult(df=df, source=str(file_path))


def load_dataset(train_path: str | Path, test_path: Optional[str | Path] = None) -> dict[str, LoadResult]:
    """
    Load train (and optionally test) datasets.

    Returns:
        dict with keys: 'train' and optionally 'test'
    """
    results = {"train": load_csv(train_path)}
    if test_path is not None:
        results["test"] = load_csv(test_path)
    return results 