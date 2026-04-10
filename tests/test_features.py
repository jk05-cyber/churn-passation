"""Tests for src/features/engineering.py."""

import pandas as pd
import pytest

from src.features.engineering import (
    add_charges_ratio,
    add_services_count,
    add_tenure_bins,
    engineer_features,
)


@pytest.fixture()
def telecom_df() -> pd.DataFrame:
    return pd.DataFrame({
        "tenure": [1, 6, 12, 24, 60],
        "MonthlyCharges": [20.0, 50.0, 70.0, 80.0, 90.0],
        "TotalCharges": [20.0, 300.0, 840.0, 1920.0, 5400.0],
        "PhoneService": ["Yes", "No", "Yes", "Yes", "No"],
        "StreamingTV": ["No", "No", "Yes", "Yes", "Yes"],
        "OnlineSecurity": ["No", "No", "No", "Yes", "Yes"],
    })


class TestAddTenureBins:
    def test_adds_column(self, telecom_df):
        result = add_tenure_bins(telecom_df)
        assert "tenure_bin" in result.columns

    def test_returns_unchanged_without_tenure(self, telecom_df):
        df = telecom_df.drop(columns=["tenure"])
        result = add_tenure_bins(df)
        assert "tenure_bin" not in result.columns

    def test_bins_within_range(self, telecom_df):
        result = add_tenure_bins(telecom_df, bins=5)
        assert result["tenure_bin"].min() >= 0
        assert result["tenure_bin"].max() < 5


class TestAddChargesRatio:
    def test_adds_column(self, telecom_df):
        result = add_charges_ratio(telecom_df)
        assert "charges_ratio" in result.columns

    def test_ratio_is_positive(self, telecom_df):
        result = add_charges_ratio(telecom_df)
        assert (result["charges_ratio"].dropna() > 0).all()

    def test_returns_unchanged_without_columns(self, telecom_df):
        df = telecom_df.drop(columns=["MonthlyCharges"])
        result = add_charges_ratio(df)
        assert "charges_ratio" not in result.columns


class TestAddServicesCount:
    def test_adds_column(self, telecom_df):
        result = add_services_count(telecom_df)
        assert "services_count" in result.columns

    def test_count_between_zero_and_num_services(self, telecom_df):
        result = add_services_count(telecom_df)
        assert result["services_count"].min() >= 0
        assert result["services_count"].max() <= 8

    def test_returns_unchanged_without_service_columns(self):
        df = pd.DataFrame({"tenure": [1, 2], "Churn": [0, 1]})
        result = add_services_count(df)
        assert "services_count" not in result.columns


class TestEngineerFeatures:
    def test_runs_all_steps(self, telecom_df):
        result = engineer_features(telecom_df)
        assert "tenure_bin" in result.columns
        assert "charges_ratio" in result.columns
        assert "services_count" in result.columns

    def test_returns_dataframe(self, telecom_df):
        result = engineer_features(telecom_df)
        assert isinstance(result, pd.DataFrame)
