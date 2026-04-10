"""Tests for src/models/train.py and src/models/evaluate.py."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.preprocessing import (
    build_preprocessor,
    encode_target,
    get_column_types,
    split_features_target,
)
from src.models.evaluate import compute_metrics, evaluate
from src.models.train import (
    build_training_pipeline,
    get_model,
    load_pipeline,
    save_pipeline,
    split_data,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(50, 8000, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Churn": rng.choice([0, 1], n),
    })


@pytest.fixture()
def split_data_fixture(sample_df):
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract"]
    X, y = split_features_target(sample_df, "Churn")
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = get_model("logistic_regression", {"logistic_regression": {"max_iter": 200}})
    pipeline = build_training_pipeline(preprocessor, model)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    return pipeline, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# get_model tests
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_logistic_regression(self):
        model = get_model("logistic_regression", {})
        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression)

    def test_random_forest(self):
        model = get_model("random_forest", {})
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)

    def test_xgboost(self):
        pytest.importorskip("xgboost")
        model = get_model("xgboost", {})
        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            get_model("svm", {})


# ---------------------------------------------------------------------------
# split_data tests
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_correct_sizes(self, sample_df):
        X, y = split_features_target(sample_df, "Churn")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05

    def test_stratified_split(self, sample_df):
        X, y = split_features_target(sample_df, "Churn")
        _, _, y_train, y_test = split_data(X, y, test_size=0.2)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.1


# ---------------------------------------------------------------------------
# train tests
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_pipeline(self, split_data_fixture):
        pipeline, X_train, _, y_train, _ = split_data_fixture
        fitted = train(pipeline, X_train, y_train, use_smote=False)
        assert isinstance(fitted, Pipeline)

    def test_pipeline_can_predict(self, split_data_fixture):
        pipeline, X_train, X_test, y_train, _ = split_data_fixture
        fitted = train(pipeline, X_train, y_train, use_smote=False)
        preprocessor = fitted.named_steps["preprocessor"]
        classifier = fitted.named_steps["classifier"]
        X_transformed = preprocessor.transform(X_test)
        preds = classifier.predict(X_transformed)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# save / load tests
# ---------------------------------------------------------------------------

class TestSaveLoadPipeline:
    def test_roundtrip(self, split_data_fixture, tmp_path):
        pipeline, X_train, _, y_train, _ = split_data_fixture
        fitted = train(pipeline, X_train, y_train, use_smote=False)

        path = str(tmp_path / "models" / "pipeline.joblib")
        save_pipeline(fitted, path)
        assert os.path.exists(path)

        loaded = load_pipeline(path)
        assert isinstance(loaded, Pipeline)

    def test_load_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(str(tmp_path / "nonexistent.joblib"))


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_known_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9])
        scores = compute_metrics(y_true, y_pred, y_prob, ["accuracy", "f1", "roc_auc"])
        assert scores["accuracy"] == pytest.approx(1.0)
        assert scores["f1"] == pytest.approx(1.0)
        assert scores["roc_auc"] == pytest.approx(1.0)

    def test_unknown_metric_skipped(self):
        y = np.array([0, 1])
        scores = compute_metrics(y, y, y.astype(float), ["unknown_metric"])
        assert "unknown_metric" not in scores


# ---------------------------------------------------------------------------
# evaluate tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_dict_with_metrics(self, split_data_fixture, tmp_path):
        pipeline, X_train, X_test, y_train, y_test = split_data_fixture
        fitted = train(pipeline, X_train, y_train, use_smote=False)

        scores = evaluate(
            fitted, X_test, y_test,
            metrics=["accuracy", "f1"],
            output_path=str(tmp_path / "eval"),
        )
        assert "accuracy" in scores
        assert "f1" in scores

    def test_creates_output_files(self, split_data_fixture, tmp_path):
        pipeline, X_train, X_test, y_train, y_test = split_data_fixture
        fitted = train(pipeline, X_train, y_train, use_smote=False)
        output_path = str(tmp_path / "eval")

        evaluate(
            fitted, X_test, y_test,
            metrics=["accuracy"],
            output_path=output_path,
        )
        assert os.path.exists(os.path.join(output_path, "confusion_matrix.png"))
        assert os.path.exists(os.path.join(output_path, "classification_report.txt"))
        assert os.path.exists(os.path.join(output_path, "metrics.csv"))
