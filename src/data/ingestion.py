"""Data ingestion module: load and validate raw churn data."""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset from *path* and return it as a DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"The data file is empty: {path}")

    logger.info("Loaded %d rows and %d columns.", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame, target_column: str, required_columns: Optional[list] = None) -> None:
    """Validate that *df* contains the expected columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    target_column:
        Name of the target (label) column.
    required_columns:
        Additional columns that must be present. Defaults to ``None``.

    Raises
    ------
    KeyError
        If any required column is missing.
    """
    missing = []

    if target_column not in df.columns:
        missing.append(target_column)

    for col in (required_columns or []):
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    logger.info("Schema validation passed.")


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Persist *df* as a CSV file at *path*.

    Creates intermediate directories if they do not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Processed data saved to %s", path)
