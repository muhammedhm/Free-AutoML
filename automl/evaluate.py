"""Hold-out evaluation of the final pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def holdout_metrics(pipeline, X_test, y_test, task: str) -> dict:
    pred = pipeline.predict(X_test)

    if task == "regression":
        return {
            "r2": float(r2_score(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
    }
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_test, proba, multi_class="ovr", labels=np.arange(proba.shape[1]))
                )
        except ValueError:
            pass  # e.g. a class is missing from the test split
    return metrics
