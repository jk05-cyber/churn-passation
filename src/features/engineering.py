"""Feature engineering module: derive and select features for churn prediction."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def add_tenure_bins(df: pd.DataFrame, tenure_col: str = "tenure", bins: int = 5) -> pd.DataFrame:
    """Add a *tenure_bin* categorical column derived from *tenure_col*.

    Divides the tenure range into *bins* equal-width intervals.  The column is
    only added when *tenure_col* is present in *df*.
    """
    if tenure_col not in df.columns:
        return df

    df = df.copy()
    df["tenure_bin"] = pd.cut(df[tenure_col], bins=bins, labels=False)
    logger.info("Added 'tenure_bin' feature.")
    return df


def add_charges_ratio(
    df: pd.DataFrame,
    monthly_col: str = "MonthlyCharges",
    total_col: str = "TotalCharges",
) -> pd.DataFrame:
    """Add a *charges_ratio* feature (MonthlyCharges / TotalCharges).

    Returns *df* unchanged when either column is absent to keep the function
    safe for datasets that do not include billing information.
    """
    if monthly_col not in df.columns or total_col not in df.columns:
        return df

    df = df.copy()
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df["charges_ratio"] = df[monthly_col] / df[total_col].replace(0, np.nan)
    logger.info("Added 'charges_ratio' feature.")
    return df


def add_services_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add *services_count* — the total number of subscribed add-on services.

    Counts columns related to streaming, security, and support services that
    are common in telecom churn datasets.  Skipped gracefully when none of the
    expected columns are present.
    """
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present = [c for c in service_cols if c in df.columns]
    if not present:
        return df

    df = df.copy()
    df["services_count"] = df[present].apply(
        lambda row: sum(1 for v in row if str(v).lower() in ("yes", "1", "true")),
        axis=1,
    )
    logger.info("Added 'services_count' feature from %d service columns.", len(present))
    return df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: str = "mean",
    random_state: int = 42,
) -> List[str]:
    """Use a Random Forest to select the most important features.

    Parameters
    ----------
    X:
        Feature matrix (already preprocessed / numeric).
    y:
        Binary target series.
    threshold:
        Importance threshold passed to
        :class:`~sklearn.feature_selection.SelectFromModel`.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    List[str]
        Column names of the selected features.
    """
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=random_state),
        threshold=threshold,
    )
    selector.fit(X, y)
    selected = list(X.columns[selector.get_support()])
    logger.info("Feature selection retained %d / %d features.", len(selected), X.shape[1])
    return selected


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to *df* in sequence.

    This is the main entry point used by the pipeline orchestrator.

    Parameters
    ----------
    df:
        Input DataFrame (raw or lightly cleaned).

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with derived features.
    """
    df = add_tenure_bins(df)
    df = add_charges_ratio(df)
    df = add_services_count(df)
    return df
