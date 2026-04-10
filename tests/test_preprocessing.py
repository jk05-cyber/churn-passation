"""Tests for src/data/ingestion.py and src/data/preprocessing.py."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import load_data, save_processed, validate_schema
from src.data.preprocessing import (
    build_preprocessor,
    drop_columns,
    encode_target,
    get_column_types,
    split_features_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "tenure": [1, 12, 24, 36, 60],
        "MonthlyCharges": [20.0, 50.0, 70.0, 80.0, 90.0],
        "TotalCharges": [20.0, 600.0, 1680.0, 2880.0, 5400.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        "Churn": ["Yes", "No", "No", "Yes", "No"],
    })


@pytest.fixture()
def csv_file(sample_df, tmp_path) -> str:
    path = str(tmp_path / "churn.csv")
    sample_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_loads_correctly(self, csv_file, sample_df):
        df = load_data(csv_file)
        assert list(df.columns) == list(sample_df.columns)
        assert len(df) == len(sample_df)

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nonexistent.csv"))

    def test_raises_for_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        pd.DataFrame().to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_data(path)


class TestValidateSchema:
    def test_passes_when_target_present(self, sample_df):
        validate_schema(sample_df, target_column="Churn")  # no exception

    def test_raises_when_target_missing(self, sample_df):
        with pytest.raises(KeyError):
            validate_schema(sample_df, target_column="NonExistent")

    def test_raises_when_required_column_missing(self, sample_df):
        with pytest.raises(KeyError):
            validate_schema(sample_df, target_column="Churn", required_columns=["missing_col"])


class TestSaveProcessed:
    def test_creates_file(self, sample_df, tmp_path):
        path = str(tmp_path / "sub" / "out.csv")
        save_processed(sample_df, path)
        assert os.path.exists(path)
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(sample_df.columns)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestDropColumns:
    def test_drops_existing_columns(self, sample_df):
        result = drop_columns(sample_df, ["tenure"])
        assert "tenure" not in result.columns

    def test_ignores_missing_columns(self, sample_df):
        result = drop_columns(sample_df, ["nonexistent"])
        assert list(result.columns) == list(sample_df.columns)


class TestEncodeTarget:
    def test_returns_binary_integers(self, sample_df):
        df, le = encode_target(sample_df, "Churn")
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_encoder_has_two_classes(self, sample_df):
        _, le = encode_target(sample_df, "Churn")
        assert len(le.classes_) == 2


class TestGetColumnTypes:
    def test_separates_numerical_and_categorical(self, sample_df):
        num, cat = get_column_types(sample_df, target_column="Churn")
        assert "tenure" in num
        assert "MonthlyCharges" in num
        assert "Contract" in cat
        assert "Churn" not in num
        assert "Churn" not in cat


class TestBuildPreprocessor:
    def test_transforms_data(self, sample_df):
        df, _ = encode_target(sample_df, "Churn")
        num_cols, cat_cols = get_column_types(df, target_column="Churn")
        X, y = split_features_target(df, "Churn")

        preprocessor = build_preprocessor(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(df)

    def test_handles_missing_values(self):
        df = pd.DataFrame({
            "num": [1.0, None, 3.0],
            "cat": ["a", None, "b"],
            "Churn": [0, 1, 0],
        })
        num, cat = get_column_types(df, "Churn")
        X, y = split_features_target(df, "Churn")
        preprocessor = build_preprocessor(num, cat)
        X_transformed = preprocessor.fit_transform(X)
        assert not np.any(np.isnan(X_transformed))


class TestSplitFeaturesTarget:
    def test_splits_correctly(self, sample_df):
        X, y = split_features_target(sample_df, "Churn")
        assert "Churn" not in X.columns
        assert y.name == "Churn"
        assert len(X) == len(y)
