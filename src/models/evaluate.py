"""Model evaluation module: compute metrics, generate reports and plots."""

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_METRIC_FNS = {
    "accuracy": lambda y, p, _: accuracy_score(y, p),
    "precision": lambda y, p, _: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, _: recall_score(y, p, zero_division=0),
    "f1": lambda y, p, _: f1_score(y, p, zero_division=0),
    "roc_auc": lambda y, _, prob: roc_auc_score(y, prob),
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: List[str],
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    metrics:
        List of metric names to compute (must be keys of ``_METRIC_FNS``).

    Returns
    -------
    Dict[str, float]
        Metric name → score.
    """
    results: Dict[str, float] = {}
    for metric in metrics:
        if metric not in _METRIC_FNS:
            logger.warning("Unknown metric '%s'; skipping.", metric)
            continue
        results[metric] = _METRIC_FNS[metric](y_true, y_pred, y_prob)
        logger.info("  %s: %.4f", metric, results[metric])
    return results


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: List[str],
    output_path: str,
) -> Dict[str, float]:
    """Run a full evaluation pass on *X_test* / *y_test*.

    Computes all requested metrics, prints the full classification report, and
    saves confusion-matrix and ROC-curve plots to *output_path*.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline with ``"preprocessor"`` and ``"classifier"``
        steps.
    X_test:
        Raw test feature DataFrame.
    y_test:
        Test target series.
    metrics:
        Metric names to report.
    output_path:
        Directory where plots and the metrics CSV are written.

    Returns
    -------
    Dict[str, float]
        Computed metric scores.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    os.makedirs(output_path, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_test)
    y_pred = classifier.predict(X_transformed)

    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_transformed)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    logger.info("=== Evaluation Results ===")
    scores = compute_metrics(y_test.values, y_pred, y_prob, metrics)

    report = classification_report(y_test, y_pred)
    logger.info("Classification Report:\n%s", report)

    report_path = os.path.join(output_path, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(report)

    metrics_df = pd.DataFrame([scores])
    metrics_df.to_csv(os.path.join(output_path, "metrics.csv"), index=False)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
    fig.savefig(os.path.join(output_path, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

    if hasattr(classifier, "predict_proba"):
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        fig.savefig(os.path.join(output_path, "roc_curve.png"), bbox_inches="tight")
        plt.close(fig)

    logger.info("Evaluation artefacts saved to %s", output_path)
    return scores
