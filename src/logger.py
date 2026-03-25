"""
Educational Goal:
- Why this module exists in an MLOps system: Centralise logging configuration
  so every module gets consistent, timestamped, leveled logs without
  each file having to set up its own handlers.
- Responsibility (separation of concerns): Configure the root logger once
  at startup. No business logic, no ML code, no file I/O beyond opening
  the log file.
- Pipeline contract (inputs and outputs): Called once from main.py at
  startup. After that, every module just does logging.getLogger(__name__)
  and gets consistent output automatically.

What this module owns:
- Dual-output logging setup (console + local file)
- Log format and timestamp style
- Log level configuration from config.yaml

What this module does NOT own:
- Any application logic
- Any ML or data processing code
- Reading config.yaml (receives log_level and log_file as arguments)
"""

import logging
import sys
from pathlib import Path


def configure_logging(
    *,
    log_level: str,
    log_file: Path,
) -> None:
    """
    Inputs:
    - log_level: Logging level string — "DEBUG", "INFO", "WARNING", "ERROR"
    - log_file: Absolute or relative path to the log file on disk

    Outputs:
    - None (configures the root logger in place)

    Why this contract matters for reliable ML delivery:
    - Dual-output logging (console + file) means developers see logs live
      in the terminal while production systems keep a persistent audit trail.
    - Calling force=True ensures this works correctly even if a module
      accidentally called basicConfig before main.py runs.
    - Zero print() statements in production code — all observability goes
      through the logger so it can be leveled, filtered, and redirected.
    """
    numeric_level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        filename=str(log_file),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[console_handler, file_handler],
        force=True,
    )