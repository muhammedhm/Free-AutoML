"""Background training jobs.

One worker thread = one training job at a time (the others wait as 'queued').
That keeps CPU/RAM usage predictable on free hosting.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from automl import run_automl

from . import store

log = logging.getLogger("automl.jobs")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="automl-train")


def submit(exp_id: str) -> None:
    _executor.submit(_run, exp_id)


def _run(exp_id: str) -> None:
    exp = store.get_experiment(exp_id)
    cfg = exp["config"]
    store.update_experiment(exp_id, status="running", started_at=time.time())

    try:
        result = run_automl(
            csv_path=str(store.dataset_path(exp["dataset_id"])),
            target=cfg["target"],
            task=cfg.get("task"),
            metric=cfg.get("metric"),
            time_budget=cfg["time_budget"],
            test_size=cfg["test_size"],
            estimators=cfg.get("estimators"),
            output_dir=str(store.EXPERIMENTS_DIR),
            run_id=exp_id,
            verbose=0,
        )
        meta = result["metadata"]
        summary = {
            "task": meta["task"],
            "metric": meta["metric"],
            "best_model": meta["best_model"],
            "cv_score": meta["cv_score"],
            "holdout_metrics": meta["holdout_metrics"],
            "total_time_s": meta["total_time_s"],
        }
        store.update_experiment(exp_id, status="completed", finished_at=time.time(), summary=summary)
    except Exception as exc:  # noqa: BLE001 - we want to report *any* failure to the user
        log.exception("Experiment %s failed", exp_id)
        store.update_experiment(
            exp_id, status="failed", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}"
        )
