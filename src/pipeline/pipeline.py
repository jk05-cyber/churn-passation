"""End-to-end ML pipeline orchestrator for churn prediction."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from src.data.ingestion import load_data, save_processed, validate_schema
from src.data.preprocessing import (
    build_preprocessor,
    drop_columns,
    encode_target,
    get_column_types,
    split_features_target,
)
from src.features.engineering import engineer_features
from src.models.evaluate import evaluate
from src.models.train import (
    build_training_pipeline,
    get_model,
    load_pipeline,
    save_pipeline,
    split_data,
    train,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML configuration file as a dict."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s", config_path)
    return cfg


class ChurnPipeline:
    """High-level orchestrator that wires together all pipeline stages.

    Parameters
    ----------
    config_path:
        Path to the ``config.yaml`` file.

    Example
    -------
    >>> pipeline = ChurnPipeline("config/config.yaml")
    >>> pipeline.run()
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.cfg = load_config(config_path)
        self._pipeline: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, float]:
        """Execute the full pipeline: ingest → preprocess → train → evaluate.

        Returns
        -------
        Dict[str, float]
            Evaluation metric scores computed on the held-out test set.
        """
        X_train, X_test, y_train, y_test = self._prepare_data()
        self._train(X_train, y_train)
        return self._evaluate(X_test, y_test)

    def run_train(self) -> None:
        """Execute only the data-preparation and training stages."""
        X_train, _, y_train, _ = self._prepare_data()
        self._train(X_train, y_train)

    def run_evaluate(self) -> Dict[str, float]:
        """Load a saved pipeline and evaluate it against the test set.

        Raises
        ------
        FileNotFoundError
            If no saved pipeline artefact is found.
        """
        _, X_test, _, y_test = self._prepare_data()
        self._pipeline = load_pipeline(self.cfg["output"]["pipeline_path"])
        return self._evaluate(X_test, y_test)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        data_cfg = self.cfg["data"]
        target = data_cfg["target_column"]

        df = load_data(data_cfg["raw_path"])
        validate_schema(df, target_column=target)

        df = drop_columns(df, data_cfg.get("drop_columns", []))
        df = engineer_features(df)
        df, _ = encode_target(df, target)

        save_processed(df, data_cfg["processed_path"])

        X, y = split_features_target(df, target)
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=data_cfg["test_size"],
            random_state=data_cfg["random_state"],
        )
        return X_train, X_test, y_train, y_test

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        model_cfg = self.cfg["model"]
        prep_cfg = self.cfg["preprocessing"]
        algorithm = model_cfg["algorithm"]

        numerical_cols, categorical_cols = get_column_types(
            X_train.assign(**{col: None for col in []}),
            target_column="__none__",
        )

        numerical_cols, categorical_cols = (
            X_train.select_dtypes(include=["int64", "float64"]).columns.tolist(),
            X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
        )

        preprocessor = build_preprocessor(
            numerical_cols,
            categorical_cols,
            numerical_strategy=prep_cfg["numerical_strategy"],
            categorical_strategy=prep_cfg["categorical_strategy"],
        )

        model = get_model(
            algorithm,
            model_cfg["hyperparameters"],
            random_state=model_cfg["random_state"],
        )

        self._pipeline = build_training_pipeline(preprocessor, model)
        self._pipeline = train(
            self._pipeline,
            X_train,
            y_train,
            use_smote=True,
            random_state=model_cfg["random_state"],
        )

        save_pipeline(self._pipeline, self.cfg["output"]["pipeline_path"])

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        eval_cfg = self.cfg["evaluation"]
        return evaluate(
            self._pipeline,
            X_test,
            y_test,
            metrics=eval_cfg["metrics"],
            output_path=eval_cfg["output_path"],
        )
