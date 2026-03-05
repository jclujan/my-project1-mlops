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
from typing import Any

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoadError(RuntimeError):
    """Raised when data cannot be loaded properly."""


@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    source: str


def load_csv(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    nrows: int | None = None,
    **read_csv_kwargs: Any,
) -> LoadResult:
    """
    Inputs:
    - path: Path to the CSV file.
    - encoding: File encoding (default utf-8).
    - nrows: If set, load only the first N rows. Useful for quick smoke-tests
      and inference on small unseen samples (e.g. nrows=20).
    - **read_csv_kwargs: Any extra keyword arguments forwarded to pd.read_csv.
    Outputs:
    - LoadResult(df, source)
    """
    file_path = Path(path)

    logger.info("Loading CSV: %s (nrows=%s)", file_path, nrows)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if not file_path.is_file():
        raise DataLoadError(f"Path is not a file: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise DataLoadError(f"Expected a .csv file, got: {file_path.suffix}")

    try:
        df = pd.read_csv(
            file_path, encoding=encoding, nrows=nrows, **read_csv_kwargs
        )
    except Exception as exc:
        logger.exception("Failed reading CSV: %s", file_path)
        raise DataLoadError(f"Failed to read CSV at {file_path}: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"CSV loaded but contains no rows: {file_path}")

    logger.info("Loaded CSV OK: %s shape=%s", file_path, df.shape)
    return LoadResult(df=df, source=str(file_path))


def load_dataset(
    train_path: str | Path,
    test_path: str | Path | None = None,
    *,
    nrows: int | None = None,
    read_csv_kwargs: dict[str, Any] | None = None,
) -> dict[str, LoadResult]:
    """
    Inputs:
    - train_path: Path to training CSV.
    - test_path: Optional path to test CSV.
    - nrows: If set, load only the first N rows from each file. Useful for
      quick pipeline smoke-tests without loading the full dataset.
    - read_csv_kwargs: Extra keyword arguments forwarded to pd.read_csv.
    Outputs:
    - dict with keys "train" (and optionally "test"), each a LoadResult.
    """
    read_csv_kwargs = read_csv_kwargs or {}
    results = {"train": load_csv(train_path, nrows=nrows, **read_csv_kwargs)}
    if test_path is not None:
        results["test"] = load_csv(test_path, nrows=nrows, **read_csv_kwargs)
    return results