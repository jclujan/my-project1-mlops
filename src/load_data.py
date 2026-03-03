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
    **read_csv_kwargs: Any,
) -> LoadResult:
    file_path = Path(path)

    logger.info("Loading CSV: %s", file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if not file_path.is_file():
        raise DataLoadError(f"Path is not a file: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise DataLoadError(f"Expected a .csv file, got: {file_path.suffix}")

    try:
        df = pd.read_csv(file_path, encoding=encoding, **read_csv_kwargs)
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
    read_csv_kwargs: dict[str, Any] | None = None,
) -> dict[str, LoadResult]:
    read_csv_kwargs = read_csv_kwargs or {}
    results = {"train": load_csv(train_path, **read_csv_kwargs)}
    if test_path is not None:
        results["test"] = load_csv(test_path, **read_csv_kwargs)
    return results