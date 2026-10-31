"""Storage layer: SQLite for records, plain folders for files.

storage/
├── automl.db
├── datasets/<ds_id>/data.csv
└── experiments/<exp_id>/{model.joblib, metadata.json, leaderboard.json, leaderboard.csv}
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(os.environ.get("AUTOML_HOME", "storage")).resolve()
DATASETS_DIR = BASE_DIR / "datasets"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
DB_PATH = BASE_DIR / "automl.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, name TEXT, n_rows INTEGER, n_cols INTEGER,
    created_at REAL, columns_json TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, dataset_id TEXT, name TEXT, status TEXT,
    config_json TEXT, error TEXT, created_at REAL, started_at REAL,
    finished_at REAL, summary_json TEXT
);
"""


def init_storage() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(r: sqlite3.Row | None) -> dict | None:
    """Row -> dict, turning every *_json column into a parsed field without the suffix."""
    if r is None:
        return None
    d = dict(r)
    for key in [k for k in d if k.endswith("_json")]:
        raw = d.pop(key)
        d[key[:-5]] = json.loads(raw) if raw else None
    return d


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------- datasets ----------------
def dataset_path(ds_id: str) -> Path:
    return DATASETS_DIR / ds_id / "data.csv"


def add_dataset(ds_id: str, name: str, n_rows: int, n_cols: int, columns: list[dict]) -> None:
    with _db() as c:
        c.execute(
            "INSERT INTO datasets VALUES (?,?,?,?,?,?)",
            (ds_id, name, n_rows, n_cols, time.time(), json.dumps(columns)),
        )


def get_dataset(ds_id: str) -> dict | None:
    with _db() as c:
        return _row(c.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone())


def list_datasets() -> list[dict]:
    with _db() as c:
        rows = c.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    return [_row(r) for r in rows]


# ---------------- experiments ----------------
def create_experiment(exp_id: str, dataset_id: str, name: str, config: dict) -> None:
    with _db() as c:
        c.execute(
            "INSERT INTO experiments (id, dataset_id, name, status, config_json, created_at) VALUES (?,?,?,?,?,?)",
            (exp_id, dataset_id, name, "queued", json.dumps(config), time.time()),
        )


def update_experiment(exp_id: str, **fields) -> None:
    allowed = {"status", "error", "started_at", "finished_at", "summary"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Cannot update fields: {unknown}")

    cols, values = [], []
    for key, val in fields.items():
        if key == "summary":
            key, val = "summary_json", json.dumps(val, default=str)
        cols.append(f"{key}=?")
        values.append(val)
    with _db() as c:
        c.execute(f"UPDATE experiments SET {', '.join(cols)} WHERE id=?", (*values, exp_id))


def get_experiment(exp_id: str) -> dict | None:
    with _db() as c:
        return _row(c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone())


def list_experiments() -> list[dict]:
    with _db() as c:
        rows = c.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    return [_row(r) for r in rows]


def fail_interrupted_experiments() -> None:
    """Jobs that were queued/running when the server stopped can never finish."""
    with _db() as c:
        c.execute(
            "UPDATE experiments SET status='failed', error='Server restarted before the job finished', "
            "finished_at=? WHERE status IN ('queued','running')",
            (time.time(),),
        )
