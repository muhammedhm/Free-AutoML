"""FastAPI backend.

Run from the project root:  uvicorn api.main:app --reload --port 8000
Interactive docs:           http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import shutil
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from automl import load_model, predict
from automl.data import clean_target, detect_task, infer_columns, load_csv
from automl.search import DEFAULT_ESTIMATORS, METRICS

from . import jobs, store
from .schemas import ExperimentRequest, PredictRequest

MAX_UPLOAD_MB = 50
MIN_ROWS = 30
MAX_PREDICT_ROWS = 50_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_storage()
    store.fail_interrupted_experiments()
    yield


app = FastAPI(title="Free AutoML API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers
def _load_df(ds_id: str) -> tuple[dict, pd.DataFrame]:
    ds = store.get_dataset(ds_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{ds_id}' not found")
    return ds, load_csv(store.dataset_path(ds_id))


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return json.loads(df.to_json(orient="records"))


def _experiment_view(exp: dict) -> dict:
    if exp["status"] == "running" and exp.get("started_at"):
        exp["elapsed_s"] = round(time.time() - exp["started_at"], 1)
    elif exp.get("started_at") and exp.get("finished_at"):
        exp["elapsed_s"] = round(exp["finished_at"] - exp["started_at"], 1)
    else:
        exp["elapsed_s"] = 0
    return exp


def _get_experiment(exp_id: str, must_be_completed: bool = False) -> dict:
    exp = store.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{exp_id}' not found")
    if must_be_completed and exp["status"] != "completed":
        raise HTTPException(409, f"Experiment is '{exp['status']}', not completed")
    return exp


def _run_dir(exp_id: str):
    return store.EXPERIMENTS_DIR / exp_id


@lru_cache(maxsize=4)
def _bundle(exp_id: str) -> dict:
    return load_model(_run_dir(exp_id))


# ------------------------------------------------------------------ meta
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    """Everything the UI needs to build its dropdowns."""
    return {"metrics": METRICS, "estimators": DEFAULT_ESTIMATORS}


# ------------------------------------------------------------------ datasets
@app.post("/datasets", status_code=201)
def upload_dataset(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")

    ds_id = store.new_id("ds")
    path = store.dataset_path(ds_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    try:
        df = load_csv(path)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(path.parent, ignore_errors=True)
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc

    columns = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "n_missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique()),
        }
        for c in df.columns
    ]
    store.add_dataset(ds_id, file.filename, len(df), df.shape[1], columns)
    return store.get_dataset(ds_id)


@app.get("/datasets")
def list_datasets():
    return store.list_datasets()


@app.get("/datasets/{ds_id}")
def get_dataset(ds_id: str):
    ds = store.get_dataset(ds_id)
    if ds is None:
        raise HTTPException(404, f"Dataset '{ds_id}' not found")
    return ds


@app.get("/datasets/{ds_id}/preview")
def preview_dataset(ds_id: str, n: int = Query(10, ge=1, le=100)):
    _, df = _load_df(ds_id)
    return {"columns": list(df.columns), "rows": _records(df.head(n))}


@app.get("/datasets/{ds_id}/profile")
def profile_dataset(ds_id: str, target: str):
    """What the AutoML engine will make of this dataset if `target` is the label."""
    ds, df = _load_df(ds_id)
    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' not found")

    df, n_dropped = clean_target(df, target)
    y = df[target]
    X = df.drop(columns=[target])
    try:
        cols = infer_columns(X)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    task = detect_task(y)
    result = {
        "n_rows": len(df),
        "n_rows_missing_target": n_dropped,
        "task": task,
        "n_unique_target": int(y.nunique()),
        "numeric_columns": cols.numeric,
        "categorical_columns": cols.categorical,
        "dropped_columns": cols.dropped,
        "warnings": [],
    }

    if task == "classification":
        counts = y.value_counts()
        result["class_counts"] = {str(k): int(v) for k, v in counts.head(30).items()}
        if counts.min() < 2:
            result["warnings"].append("Some classes have a single row; the train/test split cannot be stratified.")
        if len(counts) > 1 and counts.max() / counts.min() > 10:
            result["warnings"].append("Classes are strongly imbalanced; consider macro_f1 or roc_auc as the metric.")
        if len(counts) > 20:
            result["warnings"].append("More than 20 classes. Is this really a classification target?")
    else:
        result["target_stats"] = {k: float(v) for k, v in y.describe().items()} if y.dtype.kind in "biuf" else {}

    if len(df) < MIN_ROWS:
        result["warnings"].append(f"Only {len(df)} rows; at least {MIN_ROWS} are needed to train.")
    if cols.dropped:
        result["warnings"].append(f"{len(cols.dropped)} column(s) will be ignored (see details).")
    return result


# ------------------------------------------------------------------ experiments
@app.post("/experiments", status_code=202)
def create_experiment(req: ExperimentRequest):
    ds, df = _load_df(req.dataset_id)
    if req.target not in df.columns:
        raise HTTPException(400, f"Column '{req.target}' not found in dataset")

    df, _ = clean_target(df, req.target)
    if len(df) < MIN_ROWS:
        raise HTTPException(400, f"Need at least {MIN_ROWS} rows with a target value (found {len(df)})")

    task = req.task or detect_task(df[req.target])
    n_unique = df[req.target].nunique()
    if task == "classification" and n_unique < 2:
        raise HTTPException(400, "The target has a single value; there is nothing to learn.")
    if task == "regression" and df[req.target].dtype.kind not in "biuf":
        raise HTTPException(400, "Regression needs a numeric target column.")

    metric = req.metric or METRICS[task][0]
    if metric not in METRICS[task]:
        raise HTTPException(400, f"Metric '{metric}' is not valid for {task}. Choose from {METRICS[task]}")
    if metric == "roc_auc" and n_unique != 2:
        raise HTTPException(400, "'roc_auc' needs a binary target; use 'roc_auc_ovr' or 'macro_f1'.")
    if metric == "roc_auc_ovr" and n_unique == 2:
        raise HTTPException(400, "'roc_auc_ovr' is for multiclass targets; use 'roc_auc'.")

    estimators = req.estimators or DEFAULT_ESTIMATORS[task]
    bad = [e for e in estimators if e not in DEFAULT_ESTIMATORS[task]]
    if bad:
        raise HTTPException(400, f"Estimators {bad} are not available for {task}")

    config = {
        "target": req.target,
        "task": task,
        "metric": metric,
        "time_budget": req.time_budget,
        "test_size": req.test_size,
        "estimators": estimators,
    }
    exp_id = store.new_id("exp")
    store.create_experiment(exp_id, req.dataset_id, req.name or f"{ds['name']} -> {req.target}", config)
    jobs.submit(exp_id)
    return _experiment_view(store.get_experiment(exp_id))


@app.get("/experiments")
def list_experiments():
    return [_experiment_view(e) for e in store.list_experiments()]


@app.get("/experiments/{exp_id}")
def get_experiment(exp_id: str):
    exp = _experiment_view(_get_experiment(exp_id))
    meta_file = _run_dir(exp_id) / "metadata.json"
    if exp["status"] == "completed" and meta_file.exists():
        exp["metadata"] = json.loads(meta_file.read_text())
    return exp


@app.get("/experiments/{exp_id}/leaderboard")
def get_leaderboard(exp_id: str):
    _get_experiment(exp_id, must_be_completed=True)
    return json.loads((_run_dir(exp_id) / "leaderboard.json").read_text())


@app.get("/experiments/{exp_id}/model")
def download_model(exp_id: str):
    _get_experiment(exp_id, must_be_completed=True)
    return FileResponse(_run_dir(exp_id) / "model.joblib", filename=f"{exp_id}.joblib",
                        media_type="application/octet-stream")


# ------------------------------------------------------------------ prediction
def _predict(exp_id: str, df: pd.DataFrame, with_proba: bool) -> dict:
    _get_experiment(exp_id, must_be_completed=True)
    try:
        preds = predict(_bundle(exp_id), df, with_proba=with_proba)
    except Exception as exc:  # noqa: BLE001 - bad input should be a 422, not a 500
        raise HTTPException(422, f"Prediction failed: {exc}") from exc
    return {"n_rows": len(preds), "predictions": _records(preds)}


@app.post("/experiments/{exp_id}/predict")
def predict_json(exp_id: str, req: PredictRequest):
    return _predict(exp_id, pd.DataFrame(req.records), req.with_proba)


@app.post("/experiments/{exp_id}/predict/file")
def predict_file(exp_id: str, file: UploadFile = File(...), with_proba: bool = False):
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc
    if len(df) > MAX_PREDICT_ROWS:
        raise HTTPException(413, f"At most {MAX_PREDICT_ROWS} rows per request")
    df.columns = [str(c).strip() for c in df.columns]
    return _predict(exp_id, df, with_proba)
