"""End-to-end orchestration: CSV -> best saved pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .data import clean_target, detect_task, infer_columns, load_csv
from .evaluate import holdout_metrics
from .preprocess import build_preprocessor
from .search import build_leaderboard, clean_config, default_metric, run_search


def _feature_stats(X: pd.DataFrame, numeric: list[str], categorical: list[str], max_values: int = 30) -> dict:
    """Min/median/max for numeric columns and top values for categorical ones (used to build UI forms)."""
    stats = {}
    for c in numeric:
        s = X[c]
        stats[c] = {"type": "numeric", "min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
    for c in categorical:
        values = X[c].value_counts().head(max_values).index
        stats[c] = {"type": "categorical", "values": [v.item() if hasattr(v, "item") else v for v in values]}
    return stats


def run_automl(
    csv_path: str,
    target: str,
    task: str | None = None,          # "classification" | "regression" | None = auto-detect
    metric: str | None = None,        # None = sensible default for the task
    time_budget: int = 60,            # seconds spent searching
    test_size: float = 0.2,
    n_splits: int = 5,
    estimators: list[str] | None = None,
    output_dir: str = "artifacts",
    seed: int = 42,
    verbose: int = 1,
    run_id: str | None = None,        # folder name; auto-generated from the time if omitted
) -> dict:
    started = time.time()

    # 1. Load and clean
    df = load_csv(csv_path)
    df, n_dropped_rows = clean_target(df, target)
    y_raw = df[target]
    X = df.drop(columns=[target])

    # 2. Task + column typing
    task = task or detect_task(y_raw)
    cols = infer_columns(X)
    feature_cols = cols.numeric + cols.categorical
    X = X[feature_cols].copy()
    for c in cols.categorical:
        X[c] = X[c].astype("object")

    # 3. Target encoding (classification only)
    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw), name=target)
        n_classes = len(label_encoder.classes_)
    else:
        y = pd.to_numeric(y_raw, errors="raise").reset_index(drop=True)
        n_classes = None
    metric = metric or default_metric(task, n_classes)

    # 4. Train / hold-out split (hold-out is never seen by the search)
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    # 5. Preprocess (fit on train only, so no leakage)
    preprocessor = build_preprocessor(cols.numeric, cols.categorical)
    X_train_t = preprocessor.fit_transform(X_train)

    # 6. AutoML search
    automl = run_search(
        X_train_t, y_train, task=task, metric=metric, time_budget=time_budget,
        n_splits=n_splits, estimators=estimators, seed=seed, verbose=verbose,
    )
    leaderboard = build_leaderboard(automl, metric)

    # 7. Bundle preprocessing + best model into ONE pipeline
    pipeline = Pipeline([("preprocess", preprocessor), ("model", automl.model.estimator)])
    test_metrics = holdout_metrics(pipeline, X_test, y_test, task)

    # 8. Save artifacts
    run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "source_file": str(csv_path),
        "target": target,
        "task": task,
        "metric": metric,
        "best_model": automl.best_estimator,
        "best_config": clean_config(automl.best_config),
        "cv_score": float(leaderboard.iloc[0][f"cv_{metric}"]),
        "holdout_metrics": test_metrics,
        "feature_stats": _feature_stats(X, cols.numeric, cols.categorical),
        "numeric_columns": cols.numeric,
        "categorical_columns": cols.categorical,
        "dropped_columns": cols.dropped,
        "classes": [c.item() if hasattr(c, "item") else c for c in label_encoder.classes_]
        if label_encoder is not None else None,
        "n_rows": int(len(df)),
        "n_rows_dropped_missing_target": int(n_dropped_rows),
        "time_budget_s": time_budget,
        "total_time_s": round(time.time() - started, 1),
    }

    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder, "metadata": metadata},
                out / "model.joblib")
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    leaderboard.to_json(out / "leaderboard.json", orient="records", indent=2)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    return {"run_dir": str(out), "leaderboard": leaderboard, "metadata": metadata}
