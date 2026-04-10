"""Data preprocessing module: clean, impute, encode and scale raw data."""

import logging
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop *columns* from *df* if they are present."""
    cols_to_drop = [c for c in columns if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping columns: %s", cols_to_drop)
    return df.drop(columns=cols_to_drop, errors="ignore")


def encode_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Encode the binary target column to 0/1 integers.

    Parameters
    ----------
    df:
        Input DataFrame containing the target column.
    target_column:
        Name of the target column.

    Returns
    -------
    Tuple[pd.DataFrame, LabelEncoder]
        The DataFrame with the encoded target column and the fitted encoder.
    """
    le = LabelEncoder()
    df = df.copy()
    df[target_column] = le.fit_transform(df[target_column].astype(str))
    logger.info("Target column '%s' encoded. Classes: %s", target_column, list(le.classes_))
    return df, le


def get_column_types(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    """Return lists of numerical and categorical feature column names.

    The target column is excluded from both lists.

    Parameters
    ----------
    df:
        Input DataFrame.
    target_column:
        Name of the target column to exclude.

    Returns
    -------
    Tuple[List[str], List[str]]
        ``(numerical_columns, categorical_columns)``
    """
    feature_df = df.drop(columns=[target_column], errors="ignore")
    numerical_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["str", "object", "category", "bool"]).columns.tolist()
    logger.info("Numerical columns (%d): %s", len(numerical_cols), numerical_cols)
    logger.info("Categorical columns (%d): %s", len(categorical_cols), categorical_cols)
    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    numerical_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Build a :class:`~sklearn.compose.ColumnTransformer` for feature preprocessing.

    The transformer applies:

    * Median (or mean) imputation + :class:`~sklearn.preprocessing.StandardScaler`
      to numerical features.
    * Most-frequent imputation + :class:`~sklearn.preprocessing.OneHotEncoder`
      to categorical features.

    Parameters
    ----------
    numerical_cols:
        Names of numerical feature columns.
    categorical_cols:
        Names of categorical feature columns.
    numerical_strategy:
        Imputation strategy for numerical columns (``"median"`` or ``"mean"``).
    categorical_strategy:
        Imputation strategy for categorical columns.

    Returns
    -------
    ColumnTransformer
        The unfitted preprocessor.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numerical_strategy)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=categorical_strategy)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numerical_cols:
        transformers.append(("numerical", numerical_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split *df* into feature matrix *X* and target series *y*."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
