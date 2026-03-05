from __future__ import annotations

"""
Educational Goal:
- Centralize reusable infrastructure logic (I/O, config loading,
  filesystem safety, schema checks) so orchestration code remains clean.
- Utilities must be deterministic and side-effect explicit.
- These functions are framework-agnostic and reusable across entrypoints
  (training, batch inference, CI, CLI tools).
"""

from pathlib import Path
from typing import List

import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    """
    Inputs:
    - path: Path to config.yaml.

    Outputs:
    - Parsed configuration dictionary.

    Why this contract matters:
    - Ensures configuration is externalized from code.
    - Provides a single, testable config-loading mechanism.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Make sure you run from the project root."
        )

    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_dummy_ames_like_csv(path: Path) -> None:
    """
    Inputs:
    - path: Location where dummy CSV should be written.

    Outputs:
    - None (writes deterministic CSV).

    Why this contract matters:
    - Keeps repository runnable end-to-end for CI and onboarding.
    - Prevents hard failures when real data is not yet provided.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6, 7, 8],
            "Neighborhood": [
                "NAmes", "CollgCr", "OldTown", "NAmes",
                "Somerst", "Edwards", "NAmes", "Sawyer",
            ],
            "OverallQual": [5, 7, 4, 6, 8, 5, 6, 7],
            "YearBuilt": [1960, 2003, 1920, 1975, 2007, 1950, 1985, 1995],
            "LotArea": [8450, 9600, 11250, 9550, 14260, 14115, 10084, 10382],
            "GrLivArea": [1710, 1262, 1786, 1717, 2198, 1362, 1694, 2090],
            "SalePrice": [
                208500, 181500, 223500, 140000,
                250000, 143000, 307000, 200000,
            ],
        }
    )

    df.to_csv(path, index=False)


def fail_fast_feature_checks(
    X: pd.DataFrame,
    *,
    quantile_bin: List[str],
    categorical_onehot: List[str],
    numeric_passthrough: List[str],
) -> None:
    """
    Inputs:
    - X: Feature dataframe (post-cleaning, target removed).
    - quantile_bin: Columns expected to be numeric and binned.
    - categorical_onehot: Categorical columns for OHE.
    - numeric_passthrough: Numeric columns passed through/scaled.

    Outputs:
    - None (raises ValueError if configuration mismatch).

    Why this contract matters:
    - Prevents silent schema drift between config and data.
    - Detects incorrect dtypes before model fitting.
    - Enforces reproducible feature definitions.
    """
    configured = list(
        dict.fromkeys(quantile_bin + categorical_onehot + numeric_passthrough)
    )

    missing = [c for c in configured if c not in X.columns]
    if missing:
        raise ValueError(
            f"Feature config error: columns missing from X: {missing}. "
            f"Available: {list(X.columns)}"
        )

    non_numeric_bins = [
        c for c in quantile_bin
        if not pd.api.types.is_numeric_dtype(X[c])
    ]
    if non_numeric_bins:
        raise ValueError(
            "Feature config error: quantile_bin columns must be numeric, "
            f"but these are not: {non_numeric_bins}"
        )


def ensure_parent_dir(path: Path | str) -> None:
    """
    Inputs:
    - path: File path whose parent directory must exist.

    Outputs:
    - None (creates directory if missing).

    Why this contract matters:
    - Prevents runtime failures during artifact persistence.
    - Centralizes filesystem side effects.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)