"""Load a saved run and make predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(run_dir: str) -> dict:
    """Load model.joblib from a run folder (or a direct path to the file)."""
    p = Path(run_dir)
    p = p / "model.joblib" if p.is_dir() else p
    return joblib.load(p)


def _prepare(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    needed = meta["numeric_columns"] + meta["categorical_columns"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = df[needed].copy()
    for c in meta["categorical_columns"]:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    return X


def predict(bundle: dict, df: pd.DataFrame, with_proba: bool = False) -> pd.DataFrame:
    """Return a DataFrame with 'prediction' (and class probabilities if requested)."""
    meta, pipe, le = bundle["metadata"], bundle["pipeline"], bundle["label_encoder"]
    X = _prepare(df, meta)
    raw = pipe.predict(X)

    out = pd.DataFrame(index=df.index)
    out["prediction"] = le.inverse_transform(raw.astype(int)) if le is not None else raw

    if with_proba and le is not None and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        for i, cls in enumerate(le.classes_):
            out[f"proba_{cls}"] = proba[:, i]
    return out
