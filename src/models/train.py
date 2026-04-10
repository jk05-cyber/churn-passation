"""Model training module: build, fit and persist churn prediction models."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_SUPPORTED_ALGORITHMS = ("xgboost", "random_forest", "logistic_regression", "lightgbm")


def get_model(algorithm: str, hyperparameters: Dict[str, Any], random_state: int = 42):
    """Instantiate and return an sklearn-compatible classifier.

    Parameters
    ----------
    algorithm:
        One of ``"xgboost"``, ``"random_forest"``, ``"logistic_regression"``,
        or ``"lightgbm"``.
    hyperparameters:
        Mapping from algorithm name to a dict of constructor keyword arguments.
    random_state:
        Random seed forwarded to algorithms that support it.

    Returns
    -------
    Estimator
        An unfitted scikit-learn compatible classifier.

    Raises
    ------
    ValueError
        If *algorithm* is not supported.
    """
    params = dict(hyperparameters.get(algorithm, {}))

    if algorithm == "xgboost":
        from xgboost import XGBClassifier  # noqa: PLC0415

        params.setdefault("random_state", random_state)
        params.pop("use_label_encoder", None)
        return XGBClassifier(**params)

    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        params.setdefault("random_state", random_state)
        return LGBMClassifier(**params)

    if algorithm == "random_forest":
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)

    if algorithm == "logistic_regression":
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)

    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. Choose from: {_SUPPORTED_ALGORITHMS}"
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Returns
    -------
    Tuple
        ``(X_train, X_test, y_train, y_test)``
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split: %d train / %d test samples (%.0f%% / %.0f%%).",
        len(X_train), len(X_test),
        (1 - test_size) * 100, test_size * 100,
    )
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Over-sample the minority class with SMOTE to address class imbalance.

    Falls back gracefully when ``imbalanced-learn`` is not installed.

    Parameters
    ----------
    X_train:
        Preprocessed training feature matrix.
    y_train:
        Training target series.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Resampled ``(X_res, y_res)``.
    """
    try:
        from imblearn.over_sampling import SMOTE  # noqa: PLC0415

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE applied. New class distribution: %s",
            dict(zip(*np.unique(y_res, return_counts=True))),
        )
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not installed; skipping SMOTE.")
        return X_train, y_train


def build_training_pipeline(preprocessor, model) -> Pipeline:
    """Combine *preprocessor* and *model* into a single sklearn Pipeline."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def train(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Fit *pipeline* on the training data.

    When *use_smote* is ``True`` the preprocessor step is applied first, then
    SMOTE is applied to the transformed features before fitting the classifier.

    Parameters
    ----------
    pipeline:
        Full sklearn Pipeline with ``"preprocessor"`` and ``"classifier"`` steps.
    X_train:
        Raw training feature DataFrame.
    y_train:
        Training target series.
    use_smote:
        Whether to apply SMOTE after preprocessing.
    random_state:
        Random seed for SMOTE.

    Returns
    -------
    Pipeline
        The fitted pipeline.
    """
    if use_smote:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.fit_transform(X_train, y_train)
        X_resampled, y_resampled = apply_smote(X_transformed, y_train, random_state)
        classifier.fit(X_resampled, y_resampled)
        logger.info("Classifier fitted after SMOTE resampling.")
    else:
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline fitted without SMOTE.")

    return pipeline


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Persist *pipeline* to disk using joblib.

    Creates intermediate directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s", path)


def load_pipeline(path: str) -> Pipeline:
    """Load and return a pipeline previously saved with :func:`save_pipeline`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s", path)
    return pipeline
