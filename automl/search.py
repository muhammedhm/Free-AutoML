"""Model search with FLAML and leaderboard construction."""
from __future__ import annotations

import math

import pandas as pd
from flaml import AutoML

# FLAML reports a *loss* (lower is better). For these metrics score = 1 - loss.
_HIGHER_IS_BETTER = {"accuracy", "roc_auc", "roc_auc_ovr", "macro_f1", "micro_f1", "f1", "r2"}

DEFAULT_ESTIMATORS = {
    "classification": ["lgbm", "xgboost", "rf", "extra_tree", "lrl1"],
    "regression": ["lgbm", "xgboost", "rf", "extra_tree"],
}


# Metrics offered to users (first item = default). 'roc_auc' is binary-only, 'roc_auc_ovr' multiclass-only.
METRICS = {
    "classification": ["roc_auc", "macro_f1", "roc_auc_ovr", "micro_f1", "accuracy", "log_loss"],
    "regression": ["r2", "rmse", "mae", "mse"],
}


def clean_config(cfg: dict | None) -> dict:
    """Turn numpy scalars into plain Python values so the config is JSON-safe."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in (cfg or {}).items()}


def default_metric(task: str, n_classes: int | None = None) -> str:
    if task == "regression":
        return "r2"
    return "roc_auc" if n_classes == 2 else "macro_f1"


def loss_to_score(metric: str, loss: float) -> float:
    return 1 - loss if metric in _HIGHER_IS_BETTER else loss


def run_search(
    X_train,
    y_train,
    task: str,
    metric: str,
    time_budget: int = 60,
    n_splits: int = 5,
    estimators: list[str] | None = None,
    seed: int = 42,
    verbose: int = 1,
) -> AutoML:
    """Run FLAML cross-validated search and return the fitted AutoML object."""
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task,
        metric=metric,
        time_budget=time_budget,
        eval_method="cv",
        n_splits=n_splits,
        estimator_list=estimators or DEFAULT_ESTIMATORS[task],
        seed=seed,
        verbose=verbose,
    )
    return automl


def build_leaderboard(automl: AutoML, metric: str) -> pd.DataFrame:
    """One row per model family, ranked by best cross-validation score."""
    rows = []
    for name, loss in automl.best_loss_per_estimator.items():
        if loss is None or not math.isfinite(loss):
            continue  # this model family never finished a trial
        rows.append({
            "model": name,
            "cv_loss": loss,
            f"cv_{metric}": loss_to_score(metric, loss),
            "best_config": clean_config(automl.best_config_per_estimator.get(name)),
        })

    board = pd.DataFrame(rows).sort_values("cv_loss").reset_index(drop=True)
    board.insert(0, "rank", board.index + 1)
    return board
